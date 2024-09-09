import pandas as pd
import numpy as np

# Define the VRAM limits
vram_limits = np.arange(100, 0, -5)

# Generate random training status data for 14 models (True/False)
np.random.seed(42)  # For reproducibility
training_data = np.random.choice([True, False], size=(len(vram_limits), 14))

# Create a DataFrame
columns = ['vram_limit'] + [f'model_{i+1}' for i in range(14)]
data = pd.DataFrame(np.column_stack((vram_limits, training_data)), columns=columns)

# Save the DataFrame to a CSV file
file_path = 'vram_training_status.csv'
data.to_csv(file_path, index=False)

print(f"CSV file saved to {file_path}")