# ==============================================================================
# AUTOMATED RAG EVALUATION: CLMM & WILCOXON (BIDIRECTIONAL ONE-SIDED TESTS)
# ==============================================================================

# Install packages if missing
if(!require(ordinal)) install.packages("ordinal")
if(!require(dplyr)) install.packages("dplyr")
if(!require(tidyr)) install.packages("tidyr")
if(!require(readr)) install.packages("readr")

library(ordinal)
library(dplyr)
library(tidyr)
library(readr)

# ------------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ------------------------------------------------------------------------------

# Set input directory
input_dir <- "c:/Users/ROKADE/Documents/Joel TA/myrag/dataset/4_experiment/4c_experiment_results/dominance_test" 

# Set output directory
output_dir <- file.path(input_dir, "Statistical_Results_OneSided")
if(!dir.exists(output_dir)) dir.create(output_dir)

# Get all CSV files in the folder
csv_files <- list.files(path = input_dir, pattern = "\\.csv$", full.names = TRUE)

if(length(csv_files) < 2) {
  stop("Error: Need at least 2 CSV files to perform a comparison.")
}

cat("Found", length(csv_files), "files. Starting analysis...\n\n")

# ------------------------------------------------------------------------------
# 2. HELPER FUNCTION: READ & CLEAN DATA
# ------------------------------------------------------------------------------
process_csv <- function(filepath) {
  # Read CSV, keeping original column names
  raw_data <- read.csv(filepath, check.names = FALSE, stringsAsFactors = FALSE)
  
  # Identify the ID column
  id_col_idx <- grep("^id$", names(raw_data), ignore.case = TRUE)
  if(length(id_col_idx) == 0) stop(paste("Column 'id' not found in", basename(filepath)))
  
  # Identify Score columns
  score_cols_names <- grep(" score$", names(raw_data), ignore.case = TRUE, value = TRUE)
  score_cols_names <- score_cols_names[!grepl("reason", score_cols_names, ignore.case = TRUE)]
  
  if(length(score_cols_names) == 0) {
    warning(paste("No numeric columns ending in ' score' found in", basename(filepath)))
    return(NULL)
  }
  
  # Select ID and Score columns
  subset_data <- raw_data[, c(names(raw_data)[id_col_idx], score_cols_names)]
  colnames(subset_data)[1] <- "QuestionID"
  
  # Pivot to Long Format
  long_data <- subset_data %>%
    pivot_longer(
      cols = -QuestionID,
      names_to = "Variation_Name",
      values_to = "Score"
    ) %>%
    mutate(
      Approach = tools::file_path_sans_ext(basename(filepath)),
      Score = as.numeric(Score)
    ) %>%
    filter(Score != -1) %>% 
    na.omit()
  
  return(long_data)
}

# ------------------------------------------------------------------------------
# 3. MAIN LOOP: PAIRWISE COMPARISONS
# ------------------------------------------------------------------------------

# Generate all unique combinations (Pairs)
file_pairs <- combn(csv_files, 2, simplify = FALSE)

# Initialize summary dataframe
summary_results <- data.frame(
  Comparison = character(),
  File_A = character(),
  File_B = character(),
  
  # Wilcoxon Results
  Wilcox_P_B_gt_A = numeric(), # Is B better than A?
  Wilcox_P_A_gt_B = numeric(), # Is A better than B?
  
  # CLMM Results
  CLMM_P_B_gt_A = numeric(),
  CLMM_P_A_gt_B = numeric(),
  
  # Final Conclusion
  Winner = character(),
  stringsAsFactors = FALSE
)

for(pair in file_pairs) {
  file_A <- pair[1]
  file_B <- pair[2]
  
  name_A <- tools::file_path_sans_ext(basename(file_A))
  name_B <- tools::file_path_sans_ext(basename(file_B))
  comparison_name <- paste0(name_A, "_vs_", name_B)
  
  cat("Processing:", comparison_name, "...\n")
  
  # Load Data
  data_A <- process_csv(file_A)
  data_B <- process_csv(file_B)
  
  if(is.null(data_A) || is.null(data_B)) next
  
  # Combine Data
  combined_data <- rbind(data_A, data_B)
  
  # Factorize
  combined_data$Approach <- factor(combined_data$Approach, levels = c(name_A, name_B))
  combined_data$Score_Factor <- as.ordered(combined_data$Score)
  combined_data$QuestionID <- as.factor(combined_data$QuestionID)
  
  # Initialize output text
  output_text <- c(
    paste("=== COMPARISON REPORT:", name_A, "(A) vs", name_B, "(B) ==="),
    paste("Date:", Sys.time()),
    ""
  )
  
  # -----------------------------
  # METHOD 1: WILCOXON (One-Sided)
  # -----------------------------
  agg_data <- combined_data %>%
    group_by(QuestionID, Approach) %>%
    summarise(Mean_Score = mean(Score), .groups = 'drop') %>%
    pivot_wider(names_from = Approach, values_from = Mean_Score) %>%
    na.omit()
  
  output_text <- c(output_text, "--- 1. Paired Wilcoxon Signed-Rank Tests ---")
  
  w_p_B_gt_A <- NA
  w_p_A_gt_B <- NA
  
  if(nrow(agg_data) > 0) {
    vec_A <- agg_data[[name_A]]
    vec_B <- agg_data[[name_B]]
    
    # Test 1: Is B greater than A?
    test_B <- wilcox.test(vec_B, vec_A, paired = TRUE, alternative = "greater")
    w_p_B_gt_A <- test_B$p.value
    
    # Test 2: Is A greater than B?
    test_A <- wilcox.test(vec_A, vec_B, paired = TRUE, alternative = "greater")
    w_p_A_gt_B <- test_A$p.value
    
    output_text <- c(output_text, paste("Hypothesis 1 (B > A) P-value:", format(w_p_B_gt_A, digits=5)))
    output_text <- c(output_text, paste("Hypothesis 2 (A > B) P-value:", format(w_p_A_gt_B, digits=5)))
  } else {
    output_text <- c(output_text, "ERROR: Not enough matching data for Wilcoxon.")
  }
  
  # -----------------------------
  # METHOD 2: CLMM (One-Sided Z-Test)
  # -----------------------------
  output_text <- c(output_text, "", "--- 2. Cumulative Link Mixed Model (CLMM) ---")
  
  c_p_B_gt_A <- NA
  c_p_A_gt_B <- NA
  clmm_z <- NA
  
  tryCatch({
    # Fit model (A is reference level)
    model <- clmm(Score_Factor ~ Approach + (1 | QuestionID), data = combined_data)
    coefs <- summary(model)$coefficients
    
    # Find coefficient for B
    # Since A is ref, the coefficient name usually contains name_B
    row_idx <- grep(name_B, rownames(coefs))
    
    if(length(row_idx) > 0) {
      clmm_z <- coefs[row_idx, "z value"]
      
      # Test 1: Is B > A? (Is coefficient significantly positive?)
      # Upper tail of the Z distribution
      c_p_B_gt_A <- pnorm(clmm_z, lower.tail = FALSE)
      
      # Test 2: Is A > B? (Is coefficient significantly negative?)
      # Lower tail of the Z distribution
      c_p_A_gt_B <- pnorm(clmm_z, lower.tail = TRUE)
      
      output_text <- c(output_text, paste("Z-score for", name_B, "vs", name_A, ":", round(clmm_z, 4)))
      output_text <- c(output_text, paste("Hypothesis 1 (B > A) P-value:", format(c_p_B_gt_A, digits=5)))
      output_text <- c(output_text, paste("Hypothesis 2 (A > B) P-value:", format(c_p_A_gt_B, digits=5)))
    }
  }, error = function(e) {
    output_text <<- c(output_text, paste("CLMM Error:", e$message))
  })
  
  # -----------------------------
  # DETERMINE WINNER
  # -----------------------------
  # We use CLMM as the primary decider, falling back to Wilcoxon if CLMM failed
  final_winner <- "Tie"
  
  # Check CLMM first
  if(!is.na(c_p_B_gt_A) && !is.na(c_p_A_gt_B)) {
    if(c_p_B_gt_A < 0.05) {
      final_winner <- name_B
    } else if(c_p_A_gt_B < 0.05) {
      final_winner <- name_A
    }
  } else if(!is.na(w_p_B_gt_A)) {
    # Fallback to Wilcoxon
    if(w_p_B_gt_A < 0.05) {
      final_winner <- name_B
    } else if(w_p_A_gt_B < 0.05) {
      final_winner <- name_A
    }
  }
  
  output_text <- c(output_text, "", paste(">>> FINAL CONCLUSION:", final_winner, "<<<"))
  
  # --- SAVE REPORT ---
  writeLines(output_text, file.path(output_dir, paste0("Report_", comparison_name, ".txt")))
  
  # --- ADD TO SUMMARY ---
  summary_results <- rbind(summary_results, data.frame(
    Comparison = comparison_name,
    File_A = name_A,
    File_B = name_B,
    Wilcox_P_B_gt_A = w_p_B_gt_A,
    Wilcox_P_A_gt_B = w_p_A_gt_B,
    CLMM_P_B_gt_A = c_p_B_gt_A,
    CLMM_P_A_gt_B = c_p_A_gt_B,
    Winner = final_winner
  ))
}

# ------------------------------------------------------------------------------
# 4. WRITE FINAL SUMMARY CSV
# ------------------------------------------------------------------------------
write.csv(summary_results, file.path(output_dir, "Summary_OneSided_Results.csv"), row.names = FALSE)

cat("\nAnalysis Complete.\n")
cat("Results located in:", output_dir, "\n")