import matplotlib.pyplot as plt
import numpy as np

# Data for the runs
runs = ['WebDataset', 'MosaicML', 'FUSE_raw', 'OCIFS_raw']

# Example values for the metrics (replace these with actual values)
power_draw = {
    'mean': [650.8, 665.4, 375.7, 376.8],
    'median': [684.5, 685.5, 378.1, 377.2]
}

clocks = {
    'mean': [1696.1, 1687.4, 1951.3, 1965.0],
    'median': [1695.0, 1680.0, 1965.0, 1950.6]
}

gpu_utilization = {
    'mean': [94.8, 97.0, 98.2, 98.8],
    'median': [100.0, 100.0, 100.0, 100.0]
}

memory_utilization = {
    'mean': [60.6, 62.4, 23.5, 23.7],
    'median': [65.0, 65.0, 24.0, 24.0]
}

total_memory_used = {
    'mean': [91.5, 93.7, 94.5, 94.9],
    'median': [94.9, 94.9, 94.9, 94.9]
}

samples_per_second = [6.853, 6.869, 2.503, 2.499]  # Single value per run

# Bar width and positions
bar_width = 0.35
x = np.arange(len(runs))

# Create the figure and subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 10))

# Samples Per Second Plot
axs[0, 0].bar(x, samples_per_second, bar_width, color='purple', label='Samples/s')
axs[0, 0].set_title('Samples Per Second')
axs[0, 0].set_xticks(x)
axs[0, 0].set_xticklabels(runs)
axs[0, 0].set_ylabel('Samples/s')
axs[0, 0].legend()

# Memory Utilization Plot
axs[0, 1].bar(x - bar_width / 2, memory_utilization['mean'], bar_width, label='Mean', color='skyblue')
axs[0, 1].bar(x + bar_width / 2, memory_utilization['median'], bar_width, label='Median', color='steelblue')
axs[0, 1].set_title('Memory Draw (%)')
axs[0, 1].set_xticks(x)
axs[0, 1].set_xticklabels(runs)
axs[0, 1].set_ylabel('Utilization (%)')
axs[0, 1].legend()

# Power Draw Plot
axs[0, 2].bar(x - bar_width / 2, power_draw['mean'], bar_width, label='Mean', color='lightgreen')
axs[0, 2].bar(x + bar_width / 2, power_draw['median'], bar_width, label='Median', color='green')
axs[0, 2].set_title('Power Draw (W)')
axs[0, 2].set_xticks(x)
axs[0, 2].set_xticklabels(runs)
axs[0, 2].set_ylabel('Power (W)')
axs[0, 2].legend()

# Clocks Plot
axs[1, 0].bar(x - bar_width / 2, clocks['mean'], bar_width, label='Mean', color='salmon')
axs[1, 0].bar(x + bar_width / 2, clocks['median'], bar_width, label='Median', color='red')
axs[1, 0].set_title('Clock Speed (MHz)')
axs[1, 0].set_xticks(x)
axs[1, 0].set_xticklabels(runs)
axs[1, 0].set_ylabel('Clock Speed (MHz)')
axs[1, 0].legend()

# GPU Utilization Plot
axs[1, 1].bar(x - bar_width / 2, gpu_utilization['mean'], bar_width, label='Mean', color='skyblue')
axs[1, 1].bar(x + bar_width / 2, gpu_utilization['median'], bar_width, label='Median', color='steelblue')
axs[1, 1].set_title('GPU Utilization (%)')
axs[1, 1].set_xticks(x)
axs[1, 1].set_xticklabels(runs)
axs[1, 1].set_ylabel('Utilization (%)')
axs[1, 1].legend()

# Memory Utilization Plot
axs[1, 2].bar(x - bar_width / 2, total_memory_used['mean'], bar_width, label='Mean', color='lightpink')
axs[1, 2].bar(x + bar_width / 2, total_memory_used['median'], bar_width, label='Median', color='pink')
axs[1, 2].set_title('Memory Utilization (%)')
axs[1, 2].set_xticks(x)
axs[1, 2].set_xticklabels(runs)
axs[1, 2].set_ylabel('Utilization (%)')
axs[1, 2].legend()

fig.suptitle(f"Performance Metrics for DataLoaders in Fine-tuning 2x8 H100", fontsize=16)
# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("finetuning_perf_metrics.png", dpi=300, bbox_inches='tight')
# Show the plots
plt.show()
