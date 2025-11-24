# ==============================================================================
# STATISTICAL ANALYSIS (PER FILE / SINGLE RUN VARIATION)
# Process: Loops through every CSV, analyzes it independently, aggregates results.
# UPDATED: 
#   1. Calculates both A > B and B > A for both methods.
#   2. Logs results of Method 1 (Wilcoxon) to the text file.
# ==============================================================================

# --- 1. CONFIGURATION & SETUP ---

# !!! EDIT THIS PATH TO POINT TO YOUR CSV FOLDER !!!
INPUT_FOLDER <- "c:/Users/ROKADE/Documents/Joel TA/myrag/dataset/4_experiment/4c_experiment_results/variation_test_single"

# Output folder name
OUTPUT_FOLDER <- file.path(INPUT_FOLDER, "Statistical_Results_Per_File")

# Function to safely install and load packages
ensure_package <- function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE, quietly = TRUE)
  }
}

suppressPackageStartupMessages({
  ensure_package("tidyverse")
  ensure_package("ordinal")
  ensure_package("rstatix")
  ensure_package("tools")
})

# Validate Input Folder
if (!dir.exists(INPUT_FOLDER)) {
  stop(paste("ERROR: The input folder '", INPUT_FOLDER, "' does not exist."))
}

# Create Output Directory
if (!dir.exists(OUTPUT_FOLDER)) {
  dir.create(OUTPUT_FOLDER)
}

# --- 2. INITIALIZATION ---

csv_files <- list.files(path = INPUT_FOLDER, pattern = "\\.csv$", full.names = TRUE)

if (length(csv_files) == 0) {
  stop("No .csv files found in the specified input folder.")
}

# Initialize Master Containers to hold results from ALL files
master_results_df <- data.frame()
master_report_txt <- c()

timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
master_report_txt <- c(master_report_txt, paste("STATISTICAL ANALYSIS REPORT (PER FILE)", timestamp))
master_report_txt <- c(master_report_txt, paste("Source Data Folder:", INPUT_FOLDER))
master_report_txt <- c(master_report_txt, "=================================================\n")

# --- 3. MAIN PROCESSING LOOP ---

cat(paste("Found", length(csv_files), "files. Beginning processing...\n\n"))

for (i in seq_along(csv_files)) {
  file_path <- csv_files[i]
  filename <- basename(file_path)
  
  cat(paste0("[", i, "/", length(csv_files), "] Processing: ", filename, "...\n"))
  master_report_txt <- c(master_report_txt, paste("### FILE:", filename, "###"))
  
  # --- LOAD DATA ---
  raw_data <- read_csv(file_path, show_col_types = FALSE)
  
  # Identify columns
  potential_score_cols <- grep(" score$", names(raw_data), value = TRUE, ignore.case = TRUE)
  score_cols <- potential_score_cols[!grepl("reason", potential_score_cols, ignore.case = TRUE)]
  
  if (length(score_cols) == 0) {
    msg <- paste("   WARNING: No valid score columns found. Skipping.")
    cat(paste0(msg, "\n"))
    master_report_txt <- c(master_report_txt, msg, "-------------------------------------------------\n")
    next
  }
  
  # Extract System Names
  system_names <- gsub(" score$", "", score_cols, ignore.case = TRUE)
  
  # Identify ID/Question columns
  id_col <- names(raw_data)[grep("^id$", names(raw_data), ignore.case = TRUE)][1]
  q_col <- names(raw_data)[grep("^question$", names(raw_data), ignore.case = TRUE)][1]
  
  if (is.na(id_col) || is.na(q_col)) {
    msg <- paste("   WARNING: Missing 'id' or 'question' column. Skipping.")
    cat(paste0(msg, "\n"))
    master_report_txt <- c(master_report_txt, msg, "-------------------------------------------------\n")
    next
  }
  
  # Create clean subset
  clean_data <- raw_data %>% 
    select(all_of(c(id_col, q_col, score_cols)))
  
  colnames(clean_data)[1] <- "id"
  colnames(clean_data)[2] <- "question"
  colnames(clean_data)[3:ncol(clean_data)] <- system_names
  
  # Pivot to Long
  long_data <- clean_data %>%
    pivot_longer(
      cols = all_of(system_names),
      names_to = "System",
      values_to = "Score_Raw"
    ) %>%
    mutate(
      Score = suppressWarnings(as.numeric(as.character(Score_Raw))),
      # Factorize for Ordinal Model (0, 1, 2)
      Score_Factor = factor(Score, levels = c(0, 1, 2), ordered = TRUE),
      System = as.factor(System),
      # Unique Question ID for this file
      QuestionID = as.factor(paste0(id, "_", question)) 
    ) %>%
    drop_na(Score) %>%
    filter(Score != -1) # Remove error codes
  
  if (nrow(long_data) == 0) {
    cat("   No valid data after cleaning. Skipping.\n")
    next
  }
  
  # Identify Pairs in this specific file
  unique_systems <- levels(long_data$System)
  if(length(unique_systems) < 2) {
    cat("   Less than 2 systems found. Cannot compare.\n")
    next
  }
  
  pairs <- combn(unique_systems, 2, simplify = FALSE)
  
  # --- ANALYSIS FOR THIS FILE ---
  for (pair in pairs) {
    sys_A <- pair[1]
    sys_B <- pair[2]
    pair_name <- paste(sys_A, "vs", sys_B)
    
    master_report_txt <- c(master_report_txt, paste("   Comparison:", pair_name))
    
    # Subset and Pivot Wide for Paired Check
    pair_data <- long_data %>% filter(System %in% c(sys_A, sys_B))
    
    pair_wide <- pair_data %>%
      select(QuestionID, System, Score, Score_Factor) %>%
      pivot_wider(names_from = System, values_from = c(Score, Score_Factor))
    
    col_A_num <- paste0("Score_", sys_A)
    col_B_num <- paste0("Score_", sys_B)
    
    # Check data integrity
    if (!col_A_num %in% names(pair_wide) || !col_B_num %in% names(pair_wide)) {
      next
    }
    
    pair_wide_clean <- pair_wide %>% drop_na(all_of(c(col_A_num, col_B_num)))
    n_obs <- nrow(pair_wide_clean)
    
    if(n_obs < 5) {
      master_report_txt <- c(master_report_txt, "   WARNING: Insufficient data points (<5). Skipping.\n")
      next
    }
    
    # ============================================================
    # METHOD 1: Wilcoxon Signed-Rank Test (Raw Scores)
    # ============================================================
    vec_A <- pair_wide_clean[[col_A_num]]
    vec_B <- pair_wide_clean[[col_B_num]]
    
    # 1. Test A > B
    w_test_A_gt_B <- wilcox.test(vec_A, vec_B, paired = TRUE, alternative = "greater", exact = FALSE)
    
    # 2. Test B > A
    w_test_B_gt_A <- wilcox.test(vec_B, vec_A, paired = TRUE, alternative = "greater", exact = FALSE)
    
    # Effect Size (Calculated once, applies magnitude to both)
    eff_size_val <- NA
    try({
      eff_data_clean <- pair_wide_clean %>%
        select(QuestionID, all_of(c(col_A_num, col_B_num))) %>%
        pivot_longer(cols = c(all_of(col_A_num), all_of(col_B_num)), 
                     names_to = "Temp_System", 
                     values_to = "Score_Num") %>%
        mutate(System = ifelse(Temp_System == col_A_num, sys_A, sys_B))
      
      eff_res <- wilcox_effsize(data = eff_data_clean, Score_Num ~ System, paired = TRUE)
      eff_size_val <- eff_res$effsize
    }, silent = TRUE)
    
    # --- LOG WILCOXON TO TEXT FILE ---
    master_report_txt <- c(master_report_txt, 
                           sprintf("      [Wilcoxon] P-values: (%s>%s)=%.4f, (%s>%s)=%.4f | EffSize=%.4f", 
                                   sys_B, sys_A, w_test_B_gt_A$p.value,
                                   sys_A, sys_B, w_test_A_gt_B$p.value,
                                   ifelse(is.na(eff_size_val), 0, eff_size_val)))
    
    # --- SAVE METHOD 1: B > A ---
    master_results_df <- rbind(master_results_df, data.frame(
      File = filename,
      Comparison = pair_name,
      Method = "Wilcoxon (Paired)",
      Hypothesis_One_Sided = paste(sys_B, ">", sys_A),
      P_Value = w_test_B_gt_A$p.value,
      Metric_Val = w_test_B_gt_A$statistic,
      Effect_Size = eff_size_val
    ))
    
    # --- SAVE METHOD 1: A > B ---
    master_results_df <- rbind(master_results_df, data.frame(
      File = filename,
      Comparison = pair_name,
      Method = "Wilcoxon (Paired)",
      Hypothesis_One_Sided = paste(sys_A, ">", sys_B),
      P_Value = w_test_A_gt_B$p.value,
      Metric_Val = w_test_A_gt_B$statistic,
      Effect_Size = eff_size_val
    ))
    
    # ============================================================
    # METHOD 2: Cumulative Link Mixed Model (CLMM)
    # ============================================================
    clmm_data <- pair_data %>%
      mutate(System = factor(System, levels = c(sys_A, sys_B))) # sys_A is reference
    
    model_fit <- tryCatch({
      clmm(Score_Factor ~ System + (1 | QuestionID), data = clmm_data, 
           control = clmm.control(method = "ucminf"))
    }, error = function(e) {
      master_report_txt <<- c(master_report_txt, paste("      [CLMM Error]:", e$message))
      return(NULL)
    })
    
    if (!is.null(model_fit)) {
      sum_mod <- summary(model_fit)
      coefs <- sum_mod$coefficients
      coef_row_name <- paste0("System", sys_B)
      
      if (coef_row_name %in% rownames(coefs)) {
        estimate <- coefs[coef_row_name, "Estimate"]
        z_value <- coefs[coef_row_name, "z value"]
        
        # Stats for B > A (Reference is A, so positive estimate means B is better)
        p_B_gt_A <- pnorm(z_value, lower.tail = FALSE) 
        odds_ratio_B <- exp(estimate)
        
        # Stats for A > B (We check the lower tail of the Z distribution)
        p_A_gt_B <- pnorm(z_value, lower.tail = TRUE)
        odds_ratio_A <- exp(-estimate) # Inverse odds
        
        # --- LOG CLMM TO TEXT FILE ---
        master_report_txt <- c(master_report_txt, 
                               sprintf("      [CLMM]     P-values: (%s>%s)=%.4f, (%s>%s)=%.4f", 
                                       sys_B, sys_A, p_B_gt_A, 
                                       sys_A, sys_B, p_A_gt_B))
        
        # --- SAVE METHOD 2: B > A ---
        master_results_df <- rbind(master_results_df, data.frame(
          File = filename,
          Comparison = pair_name,
          Method = "CLMM (Ordinal Mixed)",
          Hypothesis_One_Sided = paste(sys_B, ">", sys_A),
          P_Value = p_B_gt_A,
          Metric_Val = odds_ratio_B, # Odds of B being higher
          Effect_Size = estimate
        ))
        
        # --- SAVE METHOD 2: A > B ---
        master_results_df <- rbind(master_results_df, data.frame(
          File = filename,
          Comparison = pair_name,
          Method = "CLMM (Ordinal Mixed)",
          Hypothesis_One_Sided = paste(sys_A, ">", sys_B),
          P_Value = p_A_gt_B,
          Metric_Val = odds_ratio_A, # Odds of A being higher
          Effect_Size = -estimate # Negate estimate for A perspective
        ))
      }
    }
  }
  master_report_txt <- c(master_report_txt, "-------------------------------------------------\n")
}

# --- 4. WRITE OUTPUTS ---

if (nrow(master_results_df) > 0) {
  csv_path <- file.path(OUTPUT_FOLDER, "MASTER_STATISTICAL_SUMMARY.csv")
  write.csv(master_results_df, csv_path, row.names = FALSE)
  cat(paste("Saved Master Summary CSV to:", csv_path, "\n"))
} else {
  cat("No results generated. Check input files.\n")
}

txt_path <- file.path(OUTPUT_FOLDER, "detailed_processing_log.txt")
writeLines(master_report_txt, txt_path)
cat(paste("Saved Detailed Log to:", txt_path, "\n"))

cat("Done.\n")