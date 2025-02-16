import os
import string
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def load_data(folder_path):
     """
     Load and process data from the selected folder, distinguishing between target and behavior policies.
     """
     subfolder_data = {}
     
     # Iterate over subfolders
     for subdir, dirs, _ in os.walk(folder_path):
          for subfolder in dirs:
               subfolder_path = os.path.join(subdir, subfolder)
               target_policy_files = []
               behavior_policy_files = []
               
               # Distinguish between target and behavior policy files
               for file in os.listdir(subfolder_path):
                    if "NN" in file:
                         target_policy_files.append(os.path.join(subfolder_path, file))
                    else:
                         behavior_policy_files.append(os.path.join(subfolder_path, file))
               
               # Save files by subfolder
               subfolder_data[subfolder] = {
                    "Target": target_policy_files,
                    "Behavior": behavior_policy_files
               }
     return subfolder_data

def process_files(file_list, rolling_window=10, policy_type="", subfolder_name=""):
     """
     Process files into a single DataFrame for the specified policy type.
     - For "Target" policy, keep only the first occurrence when the 'return' value changes.
     - For other files, process without additional filtering.
     """
     data_frames = []
     for file in file_list:
          # Load the CSV file
          df = pd.read_csv(file, usecols=[0, 1], header=0)  # Column A (env_step) and E (return)
          df.columns = ['env_step', 'return']
          
          # Apply additional filtering for Target Policy files (files with "NN")
          if "NN" in file and policy_type == "Target":
               df = df.loc[df['return'] != df['return'].shift()]
          
          # Apply rolling mean for smoothing
          df['return'] = df['return'].rolling(window=rolling_window, min_periods=1).mean()
          
          # Add metadata
          df['Policy'] = policy_type
          df['Subfolder'] = subfolder_name
          
          data_frames.append(df)
     
     return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()


def plot_data(df: pd.DataFrame, policy_type: str, env_name: str):
     """
     Plot data with returns against environment steps for each policy type,
     ensuring 'baseline' configurations always have the same line type and
     are positioned at the top of the legend.
     """
     # Sort 'Subfolder' column so 'baseline' configurations come first
     df['Subfolder'] = pd.Categorical(
          df['Subfolder'], 
          categories=sorted(df['Subfolder'].unique(), key=lambda x: 'baseline' not in x.lower())
     )

     plt.figure(figsize=(10, 10))

     # Plot the data
     sns.lineplot(
          data=df,
          x='env_step',
          y='return',
          hue='Subfolder',
          style='Subfolder',
          estimator='mean',
          errorbar=('se', 1)
     )

     # Update the legend order so 'baseline' is at the top
     handles, labels = plt.gca().get_legend_handles_labels()
     sorted_legend = sorted(zip(labels, handles), key=lambda x: 'baseline' not in x[0].lower())
     sorted_labels, sorted_handles = zip(*sorted_legend)
     
     # Update plot details
     plt.legend(
          sorted_handles, 
          sorted_labels, 
          title='Configurations', 
          title_fontsize=13, 
          fontsize=12, 
          loc='lower right'
     )
     plt.title(f'{policy_type} in the {env_name} Environment', fontsize=16)
     plt.xlabel('Environment Steps (Millions)', fontsize=14)
     plt.ylabel('Return', fontsize=14)
     plt.xticks(
          [0, 5_000_000, 10_000_000, 15_000_000, 20_000_000],
          ['0', '5', '10', '15', '20'],
          fontsize=12
     )
     plt.yticks(fontsize=12)
     plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
     plt.grid(True)
     plt.show()

def main():
     """
     Main function to select a folder, process data, and generate separate plots for target and behavior policies.
     """
     # Set the folder path for visualization (e.g., "Datasets/Asterix")
     folder = input("Enter the folder name (e.g., 'Datasets/Asterix'): ")
     env_name = folder.split('/')[-1]
     
     if not os.path.exists(folder):
          print(f"Folder '{folder}' does not exist. Please check the path.")
          return
     
     print(f"Processing data from: {folder}")
     subfolder_data = load_data(folder)
     
     all_target_data = []
     all_behavior_data = []
     
     # Process files for each subfolder
     for subfolder, files in subfolder_data.items():
          target_data = process_files(files['Target'], rolling_window=1, policy_type="Target", subfolder_name=subfolder)
          behavior_data = process_files(files['Behavior'], rolling_window=100, policy_type="Behavior", subfolder_name=subfolder)
          all_target_data.append(target_data)
          all_behavior_data.append(behavior_data)
     
     # Combine data across subfolders for each policy type
     target_policy_df = pd.concat(all_target_data, ignore_index=True) if all_target_data else pd.DataFrame()
     behavior_policy_df = pd.concat(all_behavior_data, ignore_index=True) if all_behavior_data else pd.DataFrame()

     # Plot target policy data
     if not target_policy_df.empty:
          plot_data(target_policy_df, policy_type='Target Policy (Greedy)', env_name=env_name)
     else:
          print("No target policy data found.")
     
     # Plot behavior policy data
     if not behavior_policy_df.empty:
          plot_data(behavior_policy_df, policy_type='Behavior Policy (E-Greedy)', env_name=env_name)
     else:
          print("No behavior policy data found.")

if __name__ == "__main__":
     main()
