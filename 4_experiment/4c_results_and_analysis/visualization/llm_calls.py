import matplotlib.pyplot as plt
import numpy as np

# --- Set up the figure style based on user request ---
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

# --- Data extracted from the Approach 2 LLM Calls table ---
judges = ['Answer Judge', 'Subgoal', 'IQ'] 
x_values = [2, 3, 4, 5]  # Corresponds to iterations

# LLM Calls (extracted from the table)
data = {
    'Answer Judge': [8.86, 10.31, 11.58, 11.98],
    'Subgoal': [9.78, 11.23, 12.9, 13.67],
    'IQ': [10.81, 13.49, 16.24, 18.16]
}

# --- Graph Generation (Line Graph) ---

# Set the figure size to 4.5 x 3.5 inches for the external legend
fig, ax = plt.subplots(figsize=(4.5, 3.5))

# Create the line plots for each judge type
for judge in judges:
    ax.plot(x_values, data[judge], label=judge, marker='o', markersize=3, linewidth=1.5)

# Set the title
ax.set_title("Approach 2 LLM Calls") 

# Add axis labels
ax.set_xlabel("Iterations")
ax.set_ylabel("Number of LLM Calls") 

# Customize the ticks and grid for readability
ax.set_xticks(x_values)
ax.set_yticks(np.arange(8.0, 19.0, 2.0)) 
ax.grid(True, linestyle='--', alpha=0.6)

# --- Place the legend outside the plot ---
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

# Adjust layout
fig.tight_layout()

# Save the plot to a PNG file
file_name = 'approach2_llm_calls_line_graph.png'
fig.savefig(file_name, dpi=300, bbox_inches='tight')

plt.close(fig)