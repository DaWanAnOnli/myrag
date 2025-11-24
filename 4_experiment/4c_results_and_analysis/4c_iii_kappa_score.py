import numpy as np
from sklearn.metrics import cohen_kappa_score

# ==========================================
# 1. CONFIGURATION: The Confusion Matrix
# ==========================================
# Modify these values to update your data.
# Rows = Human Ratings (0, 1, 2)
# Cols = LLM Ratings (0, 1, 2)

# Row 0 (Human gave 0)
H0_L0 = 27  # Human: 0, LLM: 0
H0_L1 = 4   # Human: 0, LLM: 1
H0_L2 = 0   # Human: 0, LLM: 2

# Row 1 (Human gave 1)
H1_L0 = 3   # Human: 1, LLM: 0
H1_L1 = 12  # Human: 1, LLM: 1
H1_L2 = 8   # Human: 1, LLM: 2

# Row 2 (Human gave 2)
H2_L0 = 0   # Human: 2, LLM: 0
H2_L1 = 9   # Human: 2, LLM: 1
H2_L2 = 37  # Human: 2, LLM: 2

# ==========================================
# 2. DATA RECONSTRUCTION
# ==========================================
def reconstruct_labels():
    """
    Reconstructs the individual observation lists (Human vs LLM)
    from the confusion matrix counts.
    """
    # Reconstruct Human Labels (The Ground Truth)
    # We repeat '0' for the total count of Row 0, '1' for Row 1, etc.
    row_0_count = H0_L0 + H0_L1 + H0_L2
    row_1_count = H1_L0 + H1_L1 + H1_L2
    row_2_count = H2_L0 + H2_L1 + H2_L2

    human_labels = (
        [0] * row_0_count +
        [1] * row_1_count +
        [2] * row_2_count
    )

    # Reconstruct LLM Labels (The Predictions)
    # We must map them to the exact corresponding Human entry above.
    llm_labels = (
        # Corresponding to Human Row 0
        [0] * H0_L0 + [1] * H0_L1 + [2] * H0_L2 +
        # Corresponding to Human Row 1
        [0] * H1_L0 + [1] * H1_L1 + [2] * H1_L2 +
        # Corresponding to Human Row 2
        [0] * H2_L0 + [1] * H2_L1 + [2] * H2_L2
    )

    return human_labels, llm_labels

# ==========================================
# 3. ANALYSIS & REPORT GENERATION
# ==========================================
def interpret_score(score):
    """Returns a text interpretation of the Kappa score."""
    if score < 0.00: return "Poor Agreement"
    if 0.00 <= score <= 0.20: return "Slight Agreement"
    if 0.21 <= score <= 0.40: return "Fair Agreement"
    if 0.41 <= score <= 0.60: return "Moderate Agreement"
    if 0.61 <= score <= 0.80: return "Substantial Agreement"
    if 0.81 <= score <= 1.00: return "Almost Perfect Agreement"
    return "Unknown"

def generate_analysis(human, llm):
    # 1. Standard Cohen's Kappa (Strict exact matches only)
    std_kappa = cohen_kappa_score(human, llm)

    # 2. Linear Weighted Kappa (Penalizes off-by-one less than off-by-two)
    # Ideally suited for Ordinal data (0, 1, 2)
    lin_kappa = cohen_kappa_score(human, llm, weights='linear')

    # 3. Quadratic Weighted Kappa (Heavily penalizes outliers)
    # Often used in medical datasets
    quad_kappa = cohen_kappa_score(human, llm, weights='quadratic')

    total_samples = len(human)
    
    report = f"""
=======================================================
       LLM vs HUMAN JUDGE AGREEMENT REPORT
=======================================================
Total Observations Processed: {total_samples}

-------------------------------------------------------
1. STANDARD COHEN'S KAPPA
   Score: {std_kappa:.4f}
   Interpretation: {interpret_score(std_kappa)}
   
   Note: This metric treats all errors equally. 
   (Confusing 0 with 1 is considered as bad as 0 with 2).
-------------------------------------------------------

-------------------------------------------------------
2. LINEAR WEIGHTED KAPPA (Recommended)
   Score: {lin_kappa:.4f}
   Interpretation: {interpret_score(lin_kappa)}
   
   Note: This metric accounts for ordinality. 
   It gives partial credit for being "close" (e.g. 0 vs 1).
   This is usually the best metric for ranked scoring (0-2).
-------------------------------------------------------

-------------------------------------------------------
3. QUADRATIC WEIGHTED KAPPA
   Score: {quad_kappa:.4f}
   Interpretation: {interpret_score(quad_kappa)}
   
   Note: This metric heavily penalizes extreme disagreements.
   Since your matrix has zero extreme errors (0 vs 2), 
   this score is very high.
-------------------------------------------------------

=======================================================
FINAL CONCLUSION
=======================================================
"""
    if lin_kappa > 0.6:
        report += ("The LLM demonstrates SUBSTANTIAL agreement with the human judge.\n"
                   "Based on the Linear Weighted score, the LLM is likely a \n"
                   "reliable proxy for human evaluation in this context.")
    elif lin_kappa > 0.4:
        report += ("The LLM demonstrates MODERATE agreement.\n"
                   "It may be useful for filtering, but human review is recommended \n"
                   "for critical edge cases.")
    else:
        report += ("The LLM demonstrates POOR to FAIR agreement.\n"
                   "It is NOT recommended to replace the human judge with this LLM \n"
                   "without further fine-tuning.")
                   
    return report

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        # Prepare data
        h_labels, l_labels = reconstruct_labels()
        
        # Generate report
        full_report = generate_analysis(h_labels, l_labels)
        
        # Print to console
        print(full_report)
        
        # Save to file
        filename = "kappa_report.txt"
        with open(filename, "w") as f:
            f.write(full_report)
        
        print(f"\n[SUCCESS] Full analysis saved to '{filename}'")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")