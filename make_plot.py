import os
import string
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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

def process_files(file_list, rolling_window=10, policy_type="", subfolder_name="", interp_step=1000, smooth_window=20):
     """
     Process files into a single DataFrame for the specified policy type.
     - For "Target" policy (NN files), interpolate return values on a common grid
          spanning the full [min_start, max_end] across runs, forward-filling after the last point,
          then apply light smoothing.
     - For Behavior policy, apply rolling mean smoothing directly.
     """
     data_frames = []

   # if policy_type == "Target":
        # 1) Load each run, dedupe & collect its min/max steps
     raw_dfs = []
     starts, ends = [], []
     for file in file_list:
          df = pd.read_csv(file, usecols=[0, 1], header=0)
          df.columns = ['env_step', 'return']
          df = df.drop_duplicates(subset='env_step').sort_values('env_step')
          raw_dfs.append(df)
          starts.append(df['env_step'].min())
          ends.append(df['env_step'].max())

     # 2) Build a global grid from the earliest start to the latest end
     global_start = int(min(starts))
     global_end   = int(max(ends))
     grid = np.arange(global_start, global_end + interp_step, interp_step)

     # 3) Reindex each run onto that grid, interpolate & fill-forward, then smooth
     for df in raw_dfs:
          df = df.set_index('env_step').reindex(grid)
          # linear interpolate where we have holes
          df['return'] = df['return'].interpolate(method='linear')
          # forward-fill beyond the last logged point (and backfill before the first)
          df['return'] = df['return'].ffill().bfill()
          # light centered smoothing
          df['return'] = df['return'].rolling(window=smooth_window, center=True, min_periods=1).mean()

          df = df.reset_index().rename(columns={'index': 'env_step'})
          df['Policy']    = policy_type
          df['Subfolder'] = subfolder_name
          data_frames.append(df)

#     else:
#         # Behavior policy: just rolling‐mean smooth each run
#         for file in file_list:
#             df = pd.read_csv(file, usecols=[0, 1], header=0)
#             df.columns = ['env_step', 'return']
#             df['return'] = df['return'].rolling(window=rolling_window, min_periods=1).mean()
#             df['Policy']    = policy_type
#             df['Subfolder'] = subfolder_name
#             data_frames.append(df)

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
          title_fontsize=20, 
          fontsize=18, 
          loc='lower right'
     )
     plt.title(f'{policy_type} in the {env_name} Environment', fontsize=20)
     plt.xlabel('Environment Steps (Millions)', fontsize=18)
     plt.ylabel('Return', fontsize=18)
     plt.xticks(
          [0, 5_000_000, 10_000_000, 15_000_000, 20_000_000],
          ['0', '5', '10', '15', '20'],
          fontsize=16
     )
     plt.yticks(fontsize=16)
     plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
     plt.grid(True)
     plt.show()

def main():
     """
     Main function to select a folder, process data, and generate separate plots for target and behavior policies.
     """
     # Set the folder path for visualization (e.g., datasets/ASTERIX_DATA) ")
     folder = input("Enter the folder name (e.g., 'datasets/ASTERIX_DATA'): ")
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
