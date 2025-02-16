import pandas as pd
import numpy as np
import os

def process_run(file_path):
    """
    Process a single run file to extract environment steps and returns.
    """
    try:
        df = pd.read_csv(file_path, usecols=[0, 1], header=0)  # Use columns for env_step and return
        df.columns = ['env_step', 'return']
        return df
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def summarize_metrics(folder_path, target=True):
    """
    Summarize mean and standard error of returns for all runs in a folder.
    Differentiates between target and behavior policies based on file naming.
    """
    if target:
        run_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if "NN" in file]
    else:
        run_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if "NN" not in file]
    
    metrics = []
    for file in run_files:
        df = process_run(file)
        if df is not None and not df.empty:
            metrics.append(df['return'].values[-1])  # Take the final return value
    
    if len(metrics) > 0:
        mean_return = np.mean(metrics)
        std_error = np.std(metrics) / np.sqrt(len(metrics))
        return mean_return, std_error
    else:
        return None, None

def generate_report(base_folder):
    """
    Generate a report of metrics for all configurations in the given base folder.
    """
    print(f"{'Configuration':<80} {'Target Policy Mean ± SE':<50} {'Behavior Policy Mean ± SE':<50}")
    print("-" * 180)
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        if os.path.isdir(subfolder_path):
            target_mean, target_se = summarize_metrics(subfolder_path, target=True)
            behavior_mean, behavior_se = summarize_metrics(subfolder_path, target=False)
            
            target_str = f"{target_mean:.2f} ± {target_se:.2f}" if target_mean is not None else "No Data"
            behavior_str = f"{behavior_mean:.2f} ± {behavior_se:.2f}" if behavior_mean is not None else "No Data"
            
            print(f"{subfolder:<80} {target_str:<50} {behavior_str:<50}")

def main():
    base_folder = input("Enter the base folder path containing the subfolders for configurations: ")
    if not os.path.exists(base_folder):
        print(f"Folder '{base_folder}' does not exist. Please check the path.")
        return
    generate_report(base_folder)

if __name__ == "__main__":
    main()