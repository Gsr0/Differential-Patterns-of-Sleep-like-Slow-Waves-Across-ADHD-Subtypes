import os
import re
import pandas as pd
from glob import glob


def extract_ant_data(file_path):
    """Extract behavioral indicators from a single ANT data file"""
    with open(file_path, 'r', encoding='utf-16') as f:
        content = f.read()

    # Extract all trial blocks
    trials = re.findall(r'\*\*\* LogFrame Start \*\*\*(.*?)\*\*\* LogFrame End \*\*\*', content, re.DOTALL)

    # Parse valid trials
    valid_trials = []
    for trial in trials:
        if 'Procedure: Trail' not in trial:
            continue

        trial_data = {}
        for line in trial.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                trial_data[key] = value

        if 'Target.ACC' in trial_data and 'Target.RT' in trial_data:
            valid_trials.append(trial_data)

    if not valid_trials:
        return None

    # Create dataframe
    df = pd.DataFrame(valid_trials)

    # Type conversion
    df['Target.ACC'] = pd.to_numeric(df['Target.ACC'], errors='coerce')
    df['Target.RT'] = pd.to_numeric(df['Target.RT'], errors='coerce')
    df['Cue_Type'] = df['Cue'].apply(lambda x: 'S_Cue' if 'S_Cue' in x else x)

    # Calculate overall indicators
    results = {
        'Subject': os.path.basename(file_path).split('.')[0],
        'Total_Trials': len(df),
        'ACC_Overall': df['Target.ACC'].mean() * 100,  # Convert to percentage
        'RT_Overall': df['Target.RT'].mean(),
        'RT_std': df['Target.RT'].std()
    }

    # Use only correct trials to calculate network indicators
    correct_df = df[(df['Target.ACC'] == 1) & (df['Target.RT'] > 0)].copy()

    # Calculate attention network indicators
    try:
        # Alerting Network = No_Cue RT - Double_Cue RT
        no_cue_rt = correct_df[correct_df['Cue_Type'] == 'No_Cue']['Target.RT'].mean()
        d_cue_rt = correct_df[correct_df['Cue_Type'] == 'D_Cue']['Target.RT'].mean()
        results['Alerting'] = no_cue_rt - d_cue_rt

        # Orienting Network = Center_Cue RT - Spatial_Cue RT
        center_cue_rt = correct_df[correct_df['Cue_Type'] == 'Center_Cue']['Target.RT'].mean()
        s_cue_rt = correct_df[correct_df['Cue_Type'] == 'S_Cue']['Target.RT'].mean()
        results['Orienting'] = center_cue_rt - s_cue_rt

        # Executive Control Network = Incongruent RT - Congruent RT
        inc_rt = correct_df[correct_df['Flanker'] == 'Incongruent']['Target.RT'].mean()
        con_rt = correct_df[correct_df['Flanker'] == 'Congruent']['Target.RT'].mean()
        results['Executive_Control'] = inc_rt - con_rt
    except Exception as e:
        print(f"Error calculating networks for {file_path}: {str(e)}")
        return None

    return results


def batch_process_ant_files(folder_path):
    """Batch process all ANT data files in the folder"""
    all_files = glob(os.path.join(folder_path, '*.txt'))
    all_results = []

    for file_path in all_files:
        print(f"Processing: {os.path.basename(file_path)}")
        result = extract_ant_data(file_path)
        if result:
            all_results.append(result)

    if not all_results:
        print("No valid data found in any files")
        return None

    return pd.DataFrame(all_results)


# Usage example
if __name__ == "__main__":
    # Set the folder path containing ANT data files
    data_folder = "D:\桌面\ADHD-ANT"

    # Batch process files
    results_df = batch_process_ant_files(data_folder)

    if results_df is not None:
        # Save results to CSV
        output_file = ("ANT_Results_Summary.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\nProcessed {len(results_df)} files. Results saved to {output_file}")

        # Display statistical summary
        print("\nSummary Statistics:")
        print(results_df.describe())