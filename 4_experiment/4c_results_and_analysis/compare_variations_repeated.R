# ==============================================================================
# STATISTICAL ANALYSIS FOR SYSTEM COMPARISON
# Methods: 
# 1. Aggregated Sum + Wilcoxon Signed-Rank Test
# 2. Cumulative Link Mixed Model (CLMM)
# ==============================================================================

# --- 1. CONFIGURATION & SETUP ---

# !!! EDIT THIS PATH TO POINT TO YOUR CSV FOLDER !!!
INPUT_FOLDER <- "c:/Users/ROKADE/Documents/Joel TA/myrag/dataset/4_experiment/4c_experiment_results/variation_test"

# Output folder name (will be created where the script is run)
OUTPUT_FOLDER <- file.path(INPUT_FOLDER, "Statistical_Results_OneSided")

# Function to safely install and load packages
ensure_package <- function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE, quietly = TRUE)
  }
}

# Suppress startup messages to clean up console output
suppressPackageStartupMessages({
  ensure_package("tidyverse")
  ensure_package("ordinal")
  ensure_package("rstatix")
  ensure_package("tools")
})

# Validate Input Folder
if (!dir.exists(INPUT_FOLDER)) {
  stop(paste("ERROR: The input folder '", INPUT_FOLDER, "' does not exist. Please update the INPUT_FOLDER variable."))
}

# Create Output Directory
if (!dir.exists(OUTPUT_FOLDER)) {
  dir.create(OUTPUT_FOLDER)
}

# --- 2. DATA LOADING & PROCESSING ---
cat(paste("Loading data from:", INPUT_FOLDER, "\n"))

csv_files <- list.files(path = INPUT_FOLDER, pattern = "\\.csv$", full.names = TRUE)

if (length(csv_files) == 0) {
  stop("No .csv files found in the specified input folder.")
}

all_data_long <- data.frame()

# Loop through each CSV (representing one experimental run)
for (i in seq_along(csv_files)) {
  file_path <- csv_files[i]
  
  # Read CSV
  raw_data <- read_csv(file_path, show_col_types = FALSE)
  
  # 1. Identify columns that end in " score" (case insensitive)
  potential_score_cols <- grep(" score$", names(raw_data), value = TRUE, ignore.case = TRUE)
  
  # 2. EXCLUDE columns that contain "reason" (case insensitive) to avoid the text column error
  score_cols <- potential_score_cols[!grepl("reason", potential_score_cols, ignore.case = TRUE)]
  
  if (length(score_cols) == 0) {
    warning(paste("Skipping file: No valid score columns found in", basename(file_path)))
    next
  }
  
  # Clean column names to extract System Names
  # Example: "approach_2 score" -> "approach_2"
  system_names <- gsub(" score$", "", score_cols, ignore.case = TRUE)
  
  # Select only ID, Question, and the valid Score columns
  # We allow "id" or "ID", "question" or "Question"
  # Find the actual ID/Question column names in the file
  id_col <- names(raw_data)[grep("^id$", names(raw_data), ignore.case = TRUE)][1]
  q_col <- names(raw_data)[grep("^question$", names(raw_data), ignore.case = TRUE)][1]
  
  if (is.na(id_col) || is.na(q_col)) {
     warning(paste("Skipping file", basename(file_path), "- missing 'id' or 'question' column."))
     next
  }
  
  # Create a clean subset
  clean_data <- raw_data %>% 
    select(all_of(c(id_col, q_col, score_cols)))
  
  # Standardize ID/Question names for merging later
  colnames(clean_data)[1] <- "id"
  colnames(clean_data)[2] <- "question"
  
  # Rename score columns to just system names temporarily for pivoting
  colnames(clean_data)[3:ncol(clean_data)] <- system_names
  
  # Pivot to Long Format
  # We use as.character for Score first to handle potential read errors, then convert to numeric
  long_data <- clean_data %>%
    pivot_longer(
      cols = all_of(system_names),
      names_to = "System",
      values_to = "Score_Raw"
    ) %>%
    mutate(
      Score = suppressWarnings(as.numeric(as.character(Score_Raw))), # Force numeric
      Run_ID = i,
      Original_File = basename(file_path)
    ) %>%
    select(-Score_Raw) # Remove temp column
  
  all_data_long <- bind_rows(all_data_long, long_data)
}

if (nrow(all_data_long) == 0) {
  stop("No valid data loaded. Please check your CSV files format.")
}

# --- 3. DATA CLEANING ---
cat("Cleaning data and filtering invalid scores (-1)...\n")

# 1. Filter out -1 scores (Errors) and NAs
# 2. Convert Score to Ordered Factor (Crucial for CLMM)
clean_df <- all_data_long %>%
  drop_na(Score) %>%
  filter(Score != -1) %>%
  mutate(
    Score_Num = Score, 
    # Ensure factor has all levels 0, 1, 2 even if some are missing in data
    Score_Factor = factor(Score, levels = c(0, 1, 2), ordered = TRUE),
    System = as.factor(System),
    QuestionID = as.factor(paste0(id, "_", question)) 
  )

# Identify all unique systems
systems <- levels(clean_df$System)
pairs <- combn(systems, 2, simplify = FALSE)

# Initialize results containers
results_txt <- c()
results_df <- data.frame()

timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
results_txt <- c(results_txt, paste("STATISTICAL ANALYSIS REPORT", timestamp))
results_txt <- c(results_txt, paste("Source Data Folder:", INPUT_FOLDER))
results_txt <- c(results_txt, "=================================================\n")

# --- 4. ANALYSIS LOOP ---
for (pair in pairs) {
  sys_A <- pair[1]
  sys_B <- pair[2]
  
  pair_name <- paste(sys_A, "vs", sys_B)
  cat(paste("Analyzing:", pair_name, "...\n"))
  results_txt <- c(results_txt, paste("--- COMPARISON:", pair_name, "---"))
  
  # Subset data for this pair only
  pair_data <- clean_df %>% 
    filter(System %in% c(sys_A, sys_B))
  
  # Reshape to wide to filter incomplete pairs (Must have A and B for the same Q and Run)
  pair_wide <- pair_data %>%
    select(QuestionID, Run_ID, System, Score_Num, Score_Factor) %>%
    pivot_wider(names_from = System, values_from = c(Score_Num, Score_Factor))
  
  col_A_num <- paste0("Score_Num_", sys_A)
  col_B_num <- paste0("Score_Num_", sys_B)
  
  # Check if columns exist (in case one system is totally empty)
  if (!col_A_num %in% names(pair_wide) || !col_B_num %in% names(pair_wide)) {
    results_txt <- c(results_txt, "WARNING: One system has no valid data for this pair.\n")
    next
  }
  
  # Drop rows where one system is missing
  pair_wide_clean <- pair_wide %>%
    drop_na(all_of(c(col_A_num, col_B_num)))
  
  n_obs <- nrow(pair_wide_clean)
  results_txt <- c(results_txt, paste("Valid paired observations (Questions * Runs):", n_obs))
  
  if(n_obs < 10) {
    results_txt <- c(results_txt, "WARNING: Insufficient data points (<10) for valid statistical testing.\n")
    next
  }

  # ============================================================
  # METHOD 1: Wilcoxon Signed-Rank Test (Aggregated Sums)
  # ============================================================
  results_txt <- c(results_txt, "\n[METHOD 1: Aggregated Wilcoxon Signed-Rank Test]")
  
  # Aggregate sums per question
  agg_data <- pair_data %>%
    filter(paste(QuestionID, Run_ID) %in% paste(pair_wide_clean$QuestionID, pair_wide_clean$Run_ID)) %>%
    group_by(QuestionID, System) %>%
    summarise(Summed_Score = sum(Score_Num), .groups = 'drop') %>%
    pivot_wider(names_from = System, values_from = Summed_Score)
  
  vec_A <- agg_data[[sys_A]]
  vec_B <- agg_data[[sys_B]]
  
  w_test_A_gt_B <- wilcox.test(vec_A, vec_B, paired = TRUE, alternative = "greater")
  w_test_B_gt_A <- wilcox.test(vec_B, vec_A, paired = TRUE, alternative = "greater")
  
  eff_size <- wilcox_effsize(
    data = agg_data %>% pivot_longer(cols = c(all_of(sys_A), all_of(sys_B)), names_to="System", values_to="Summed"),
    formula = Summed ~ System,
    paired = TRUE
  )
  
  results_txt <- c(results_txt, sprintf("   Hypothesis (%s > %s): p-value = %.5f", sys_A, sys_B, w_test_A_gt_B$p.value))
  results_txt <- c(results_txt, sprintf("   Hypothesis (%s > %s): p-value = %.5f", sys_B, sys_A, w_test_B_gt_A$p.value))
  
  results_df <- rbind(results_df, data.frame(
    Comparison = pair_name,
    Method = "Wilcoxon (Summed)",
    Hypothesis_One_Sided = paste(sys_A, ">", sys_B),
    P_Value = w_test_A_gt_B$p.value,
    Metric_Value = w_test_A_gt_B$statistic,
    Metric_Name = "V-statistic",
    Effect_Size = eff_size$effsize
  ))
  results_df <- rbind(results_df, data.frame(
    Comparison = pair_name,
    Method = "Wilcoxon (Summed)",
    Hypothesis_One_Sided = paste(sys_B, ">", sys_A),
    P_Value = w_test_B_gt_A$p.value,
    Metric_Value = w_test_B_gt_A$statistic,
    Metric_Name = "V-statistic",
    Effect_Size = eff_size$effsize
  ))

  # ============================================================
  # METHOD 2: Cumulative Link Mixed Model (CLMM)
  # ============================================================
  results_txt <- c(results_txt, "\n[METHOD 2: Cumulative Link Mixed Model (CLMM)]")
  
  clmm_data <- pair_wide_clean %>%
    pivot_longer(cols = starts_with("Score_Factor"), names_to = "Sys_Col", values_to = "Score_Factor") %>%
    mutate(System = ifelse(Sys_Col == paste0("Score_Factor_", sys_A), sys_A, sys_B)) %>%
    mutate(System = factor(System, levels = c(sys_A, sys_B)))
  
  model_fit <- tryCatch({
    clmm(Score_Factor ~ System + (1 | QuestionID), data = clmm_data)
  }, error = function(e) {
    results_txt <<- c(results_txt, paste("   ERROR: CLMM failed to converge:", e$message))
    return(NULL)
  })
  
  if (!is.null(model_fit)) {
    sum_mod <- summary(model_fit)
    coefs <- sum_mod$coefficients
    coef_row_name <- paste0("System", sys_B)
    
    if (coef_row_name %in% rownames(coefs)) {
      estimate <- coefs[coef_row_name, "Estimate"]
      z_value <- coefs[coef_row_name, "z value"]
      odds_ratio <- exp(estimate)
      
      p_B_gt_A <- pnorm(z_value, lower.tail = FALSE)
      p_A_gt_B <- pnorm(z_value, lower.tail = TRUE)
      
      results_txt <- c(results_txt, sprintf("   Odds Ratio (%s vs %s): %.4f", sys_B, sys_A, odds_ratio))
      results_txt <- c(results_txt, sprintf("   Hypothesis (%s > %s): p-value = %.5f", sys_B, sys_A, p_B_gt_A))
      results_txt <- c(results_txt, sprintf("   Hypothesis (%s > %s): p-value = %.5f", sys_A, sys_B, p_A_gt_B))
      
      results_df <- rbind(results_df, data.frame(
        Comparison = pair_name,
        Method = "CLMM (Ordinal Mixed)",
        Hypothesis_One_Sided = paste(sys_B, ">", sys_A),
        P_Value = p_B_gt_A,
        Metric_Value = odds_ratio,
        Metric_Name = "Odds Ratio (B/A)",
        Effect_Size = estimate
      ))
      results_df <- rbind(results_df, data.frame(
        Comparison = pair_name,
        Method = "CLMM (Ordinal Mixed)",
        Hypothesis_One_Sided = paste(sys_A, ">", sys_B),
        P_Value = p_A_gt_B,
        Metric_Value = 1/odds_ratio,
        Metric_Name = "Odds Ratio (A/B)",
        Effect_Size = -estimate 
      ))
    }
  }
  results_txt <- c(results_txt, "\n-------------------------------------------------\n")
}

# --- 5. WRITE OUTPUTS ---
txt_path <- file.path(OUTPUT_FOLDER, "statistical_report.txt")
writeLines(results_txt, txt_path)

csv_path <- file.path(OUTPUT_FOLDER, "statistical_summary.csv")
write.csv(results_df, csv_path, row.names = FALSE)

cat(paste("\nAnalysis Complete. Results saved in folder:", OUTPUT_FOLDER, "\n"))