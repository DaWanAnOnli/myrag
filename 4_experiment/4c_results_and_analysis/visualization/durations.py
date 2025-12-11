import matplotlib.pyplot as plt
import numpy as np

# --- Set up the figure style based on user request ---
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

# --- Data extracted from the Approach 2 Multi-Pipeline table ---
pipelines = ['Both (attempt 1)', 'Both (attempt 2)', 'Router', 'Query Decomposer']
duration_values = [131.36, 85.63, 10.91, 44.59]

# --- Graph Generation (Bar Chart) ---

# Set the figure size to 3.5 x 3.5 inches
fig, ax = plt.subplots(figsize=(3.5, 3.5)) 

# Create a bar chart
x_pos = np.arange(len(pipelines))
colors = ['#f59e0b', '#fcd34d', '#10b981', '#3b82f6']
bars = ax.bar(x_pos, duration_values, color=colors, width=0.7) 

# Set the title
ax.set_title("Approach 2 Multi-Pipeline Median Duration") 

# Add axis labels
ax.set_xlabel("Pipeline Component")
ax.set_ylabel("Median Duration (seconds)") 

# Customize the ticks and grid for readability
ax.set_xticks(x_pos)
ax.set_xticklabels(pipelines, rotation=20, ha='right')
ax.set_yticks(range(0, 150, 25)) 
ax.grid(axis='y', linestyle='--', alpha=0.6)


# Adjust layout
fig.tight_layout()

# Save the plot to a PNG file
file_name = 'approach2_multi_pipeline_median_duration_bar_chart.png'
fig.savefig(file_name, dpi=300, bbox_inches='tight')

plt.close(fig)