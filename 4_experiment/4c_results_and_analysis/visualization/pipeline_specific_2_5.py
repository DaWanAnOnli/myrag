import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up the figure style for publication
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

# --- DATA EXTRACTION (Approach 2 Query Decomposer) ---
# Overall Metrics: {0B16C5C7...}.png (Approach 2 Query Decomposer)
# Top 5 USED decisions: {0B16C5C7...}.png (Approach 2 Query Decomposer, bottom part)
# Top 5 PERFORMING decisions: {68A77540...}.png (Top 5 performing decomposer decision)

# 1. Overall Score Distribution Data - Extracted from {0B16C5C7...}.png
incorrect_overall = 42.24
partial_overall = 27.84
correct_overall = 29.92
duration = 44.78
llm_calls = 7.15

# 2. Top 5 USED Decomposer Decisions - Extracted from {0B16C5C7...}.png
used_labels = ['1N_primary', '1G_primary, 1N_support', '1G_support_1N_primary', '2N_primary', '1G_primary_1N_support'] # Simplified label 5 due to data mismatch
used_values = [42.64, 15.76, 13.52, 10.40, 3.52]

# Score Distribution by Top 5 USED Decomposer Decisions
used_score0 = [45.59, 34.52, 30.77, 49.23, 47.73]
used_score1 = [20.45, 40.10, 30.18, 26.15, 31.82]
used_score2 = [33.96, 25.38, 39.05, 24.62, 20.45]

# 3. Top 5 PERFORMING Decomposer Decisions - Extracted from {68A77540...}.png
# Note: The "Top 5 Performing" table in {68A77540...}.png contains decisions that are also present in the "Top 5 Used" table, but the metric is different (performance vs. usage). The labels were extracted in the previous turn.
performing_labels = ['1G_support, 2N_primary', '1G_support_1N_primary', '1G_primary_1N_support', '1N_primary', '2G_primary_2N_support']
performing_values = [2.40, 13.52, 15.76, 42.64, 0.56] # % from total

# Score Distribution by Top 5 PERFORMING Decomposer Decisions
performing_score0 = [30.00, 30.77, 34.52, 45.59, 28.57]
performing_score1 = [26.67, 30.18, 40.10, 20.45, 57.14]
performing_score2 = [43.33, 39.05, 25.38, 33.96, 14.29]

# --- PLOTTING CONFIGURATION (3.5x3.5, External Legend) ---
SQUARE_FIGSIZE = (3.5, 3.5)
FILE_SUFFIX = "_approach2_decomposer_5charts_3_5.png"
LEGEND_KWARGS = {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1)}
width = 0.2
bar_width = 0.8 / 3

# --- 1. Overall Score Distribution - Bar Chart ---
fig1, ax1 = plt.subplots(figsize=SQUARE_FIGSIZE)
labels = ['Total']
incorrect = [incorrect_overall]
partial = [partial_overall]
correct = [correct_overall]
x = np.arange(len(labels))

ax1.bar(x - width, incorrect, width, label='Incorrect', color='#ef4444')
ax1.bar(x, partial, width, label='Partially Correct', color='#f59e0b')
ax1.bar(x + width, correct, width, label='Correct', color='#10b981')

ax1.set_xlabel('Approach')
ax1.set_ylabel('Percentage (%)')
ax1.set_title('Overall Score Distribution')
ax1.set_xticks(x)
ax1.set_xticklabels(['Combined Result'])
ax1.set_ylim(0, 100)
ax1.legend(ncol=1, **LEGEND_KWARGS)
ax1.grid(axis='y', alpha=0.3)
fig1.tight_layout()
fig1.savefig('chart1_overall_score_distribution' + FILE_SUFFIX, dpi=300, bbox_inches='tight')
plt.close(fig1)

# --- 2. Top 5 USED Decomposer Decision Distribution - Bar Chart ---
fig2, ax2 = plt.subplots(figsize=SQUARE_FIGSIZE)
ax2.bar(used_labels, used_values, color=['#3b82f6', '#8b5cf6', '#ec4899', '#14b8a6', '#f472b6'])

ax2.set_ylabel('Percentage (%)')
ax2.set_title('Top 5 Used Decomposer Decision Distribution')
ax2.set_ylim(0, 50)
ax2.grid(axis='y', alpha=0.3)
plt.xticks(rotation=40, ha='right')
fig2.tight_layout()
fig2.savefig('chart2_used_decision_distribution' + FILE_SUFFIX, dpi=300, bbox_inches='tight')
plt.close(fig2)

# --- 3. Score Distribution by Top 5 USED Decomposer Decisions - Grouped Bar Chart ---
fig3, ax3 = plt.subplots(figsize=SQUARE_FIGSIZE)
x = np.arange(len(used_labels))

ax3.bar(x - bar_width, used_score0, bar_width, label='Incorrect', color='#ef4444')
ax3.bar(x, used_score1, bar_width, label='Partially Correct', color='#f59e0b')
ax3.bar(x + bar_width, used_score2, bar_width, label='Correct', color='#10b981')

ax3.set_xlabel('Decomposer Decision')
ax3.set_ylabel('Percentage (%)')
ax3.set_title('Score Distribution by Top 5 Used Decisions')
ax3.set_xticks(x)
ax3.set_xticklabels(used_labels, rotation=40, ha='right')
ax3.legend(ncol=1, **LEGEND_KWARGS)
ax3.grid(axis='y', alpha=0.3)
fig3.tight_layout()
fig3.savefig('chart3_used_score_distribution' + FILE_SUFFIX, dpi=300, bbox_inches='tight')
plt.close(fig3)

# --- 4. Top 5 PERFORMING Decomposer Decision Distribution - Bar Chart ---
fig4, ax4 = plt.subplots(figsize=SQUARE_FIGSIZE)
ax4.bar(performing_labels, performing_values, color=['#3b82f6', '#8b5cf6', '#ec4899', '#14b8a6', '#f472b6'])

ax4.set_ylabel('Percentage (%)')
ax4.set_title('Top 5 Performing Decomposer Decision Distribution')
ax4.set_ylim(0, 50)
ax4.grid(axis='y', alpha=0.3)
plt.xticks(rotation=40, ha='right')
fig4.tight_layout()
fig4.savefig('chart4_performing_decision_distribution' + FILE_SUFFIX, dpi=300, bbox_inches='tight')
plt.close(fig4)

# --- 5. Score Distribution by Top 5 PERFORMING Decomposer Decisions - Grouped Bar Chart ---
fig5, ax5 = plt.subplots(figsize=SQUARE_FIGSIZE)
x = np.arange(len(performing_labels))

ax5.bar(x - bar_width, performing_score0, bar_width, label='Incorrect', color='#ef4444')
ax5.bar(x, performing_score1, bar_width, label='Partially Correct', color='#f59e0b')
ax5.bar(x + bar_width, performing_score2, bar_width, label='Correct', color='#10b981')

ax5.set_xlabel('Decomposer Decision')
ax5.set_ylabel('Percentage (%)')
ax5.set_title('Score Distribution by Top 5 Performing Decisions')
ax5.set_xticks(x)
ax5.set_xticklabels(performing_labels, rotation=45, ha='right')
ax5.legend(ncol=1, **LEGEND_KWARGS)
ax5.grid(axis='y', alpha=0.3)
fig5.tight_layout()
fig5.savefig('chart5_performing_score_distribution' + FILE_SUFFIX, dpi=300, bbox_inches='tight')
plt.close(fig5)

print("Five unique graphs for the 'Approach 2 Query Decomposer' data have been generated at 3.5x3.5 inches with external legends.")
print("Files created: chart1_overall_score_distribution_approach2_decomposer_5charts_3_5.png, chart2_used_decision_distribution_approach2_decomposer_5charts_3_5.png, chart3_used_score_distribution_approach2_decomposer_5charts_3_5.png, chart4_performing_decision_distribution_approach2_decomposer_5charts_3_5.png, chart5_performing_score_distribution_approach2_decomposer_5charts_3_5.png")