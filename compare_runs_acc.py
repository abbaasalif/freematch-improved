import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files into DataFrames
data_original = pd.read_csv('run_data/run-freematch_cifar10_4000_original_tensorboard_runs_cifar_10_4000_run_0-tag-best_acc.csv')
data_mod1 = pd.read_csv('run_data/run-freematch_cifar10_4000_modification1_tensorboard_runs_cifar_10_4000_run_0-tag-best_acc.csv')
data_mod2 = pd.read_csv('run_data/run-freematch_cifar10_4000_modification_2_tensorboard_runs_cifar_10_4000_1_run_0-tag-best_acc.csv')
data_mod3 = pd.read_csv('run_data/run-freematch_cifar10_4000_modification_3_tensorboard_runs_cifar_10_4000_2_run_0-tag-best_acc.csv')

# # Plot each data set
# plt.figure(figsize=(10, 6))

# plt.plot(data_original['Step'], data_original['Value'], label='Original', marker='o', color='blue', linestyle='-')
# plt.plot(data_mod1['Step'], data_mod1['Value'], label='Modification 1', marker='^', color='green', linestyle='--')
# plt.plot(data_mod2['Step'], data_mod2['Value'], label='Modification 2', marker='s', color='red', linestyle='-.')
# plt.plot(data_mod3['Step'], data_mod3['Value'], label='Modification 3', marker='*', color='purple', linestyle=':')

# plt.title('Comparison of CIFAR-10 Best Accuracy Across Modifications')
# plt.xlabel('Step')
# plt.ylabel('Best Accuracy')
# plt.legend()
# plt.grid(True)

# # Save plot to file
# plt.savefig('comparison_chart_best_acc.png')

# # Show plot
# plt.show()

# Compare the best accuracy values over the interval where Step is 30k to last step

# Plot each data set
plt.figure(figsize=(10, 6))

plt.plot(data_original[data_original['Step']>=30000]['Step'], data_original[data_original['Step']>=30000]['Value'], label='Original', marker='o', color='blue', linestyle='-')
plt.plot(data_mod1[data_mod1['Step']>=30000]['Step'], data_mod1[data_mod1['Step']>=30000]['Value'], label='Modification 1', marker='^', color='green', linestyle='--')
plt.plot(data_mod2[data_mod2['Step']>=30000]['Step'], data_mod2[data_mod2['Step']>=30000]['Value'], label='Modification 2', marker='s', color='red', linestyle='-.')
plt.plot(data_mod3[data_mod3['Step']>=30000]['Step'], data_mod3[data_mod3['Step']>=30000]['Value'], label='Modification 3', marker='*', color='purple', linestyle=':')

plt.title('Comparison of CIFAR-10 Best Accuracy Across Modifications')
plt.xlabel('Step')
plt.ylabel('Best Accuracy')
plt.legend()
plt.grid(True)

# Save plot to file
plt.savefig('comparison_chart_best_acc_zoomed.png')

# Show plot
plt.show()
