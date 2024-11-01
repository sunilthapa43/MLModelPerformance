import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = 'model_results_on_ton_iot.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Round the specified metrics to 2 decimal places without scaling
metrics = ['F1 Score', 'Accuracy', 'FPR']
data[metrics] = data[metrics].round(4)  # Round each metric to 4 decimal places to retain precision

models = data['Model']
n_metrics = len(metrics)

# Define bar width and positions
bar_width = 0.2
index = np.arange(len(models))

# Plot each metric
fig, ax = plt.subplots(figsize=(12, 8))

for i, metric in enumerate(metrics):
    bars = plt.bar(index + i * bar_width, data[metric], bar_width, label=metric)

    # Add labels on top of each bar with rounded values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')

# Set title and labels
plt.title('Performance Evaluation of Models')
plt.xlabel('Models')
plt.ylabel('Scores')
plt.xticks(index + bar_width, models, rotation=45)
plt.legend(loc='upper right')

# Set y-axis limit slightly below 1.0
plt.ylim(0, 0.999)  # Set upper limit to 0.999 to avoid showing 1

# Display and save plot
plt.tight_layout()
plt.savefig('model_performance_on_ton_iot_plot.png', dpi=300)  # Replace with your desired file path
plt.show()
