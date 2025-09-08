import pandas as pd


def analyze_ant_data(file_path='lht.txt'):
    """
    Analyze ANT task log files to extract behavioral indicators and attention network effect values.

    Parameters:
    file_path (str): Path to the ANT task log file (.txt format).

    Returns:
    dict: Dictionary containing all calculated indicators.
    """
    try:
        # Load data using tab as separator
        df = pd.read_csv(file_path, sep='\t', on_bad_lines='warn')
        print(f"File '{file_path}' loaded successfully, containing {len(df)} rows of raw data.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please ensure the file path is correct.")
        return None

    # --- 1. Data preprocessing ---

    # Filter out formal experiment trials (exclude practice 'PracProc')
    df_exp = df[df['Procedure'] == 'TrialProc'].copy()
    if df_exp.empty:
        print("Error: No formal experiment trials found in the file (Procedure == 'TrialProc'). Please check file content.")
        return None

    print(f"Filtered {len(df_exp)} formal experiment trials.")

    # Convert reaction time column to numeric type, set to NaN on error
    df_exp['Target.RT'] = pd.to_numeric(df_exp['Target.RT'], errors='coerce')

    # Clean data: remove trials with reaction times too fast (<100ms) or too slow (>1500ms)
    # Note: Target.RT == 0 indicates a miss, which we keep for now
    valid_rt_mask = (df_exp['Target.RT'] >= 100) & (df_exp['Target.RT'] <= 1500)
    # Only apply this filter when there is a reaction time
    responded_trials = df_exp[df_exp['Target.RT'] > 0].copy()
    responded_trials_cleaned = responded_trials[
        (responded_trials['Target.RT'] >= 100) & (responded_trials['Target.RT'] <= 1500)
        ]

    miss_trials = df_exp[df_exp['Target.RT'] == 0]

    print(f"Among trials with responses, removed {len(responded_trials) - len(responded_trials_cleaned)} abnormal reaction time trials.")

    # --- 2. Calculate overall behavioral indicators ---

    total_trials = len(df_exp)
    num_responded = len(responded_trials_cleaned)
    num_miss = len(miss_trials) + (len(responded_trials) - len(responded_trials_cleaned))  # Misses + removed abnormal trials

    # Calculate accuracy and error rate only in responded, cleaned trials
    correct_trials_df = responded_trials_cleaned[responded_trials_cleaned['Target.ACC'] == 1]
    num_correct = len(correct_trials_df)
    num_error = num_responded - num_correct

    # Calculate indicators
    # Accuracy = Correct count / (Correct count + Error count)
    accuracy = num_correct / num_responded if num_responded > 0 else 0
    # Error rate = Error count / (Correct count + Error count)
    error_rate = num_error / num_responded if num_responded > 0 else 0
    # Miss rate = Miss count / Total trial count
    miss_rate = num_miss / total_trials if total_trials > 0 else 0
    # Mean reaction time (correct trials only)
    mean_rt_correct = correct_trials_df['Target.RT'].mean()

    # --- 3. Calculate attention network effect values ---

    # Define function to calculate mean reaction time for each condition (correct trials only)
    def get_condition_rt(df_correct, cue_type=None, flank_type=None):
        df_slice = df_correct
        if cue_type:
            df_slice = df_slice[df_slice['CueType'] == cue_type]
        if flank_type:
            df_slice = df_slice[df_slice['FlankType'] == flank_type]
        return df_slice['Target.RT'].mean()

    # Calculate mean RT for each condition
    rt_no_cue = get_condition_rt(correct_trials_df, cue_type='nocue')
    rt_double_cue = get_condition_rt(correct_trials_df, cue_type='double')
    rt_center_cue = get_condition_rt(correct_trials_df, cue_type='center')
    rt_spatial_cue = get_condition_rt(correct_trials_df, cue_type='spatial')
    rt_congruent = get_condition_rt(correct_trials_df, flank_type='congruent')
    rt_incongruent = get_condition_rt(correct_trials_df, flank_type='incongruent')

    # Calculate network effects
    # Alerting effect = No cue RT - Double cue RT
    alerting_effect = rt_no_cue - rt_double_cue
    # Orienting effect = Center cue RT - Spatial cue RT
    orienting_effect = rt_center_cue - rt_spatial_cue
    # Executive control (conflict) effect = Incongruent RT - Congruent RT
    executive_control_effect = rt_incongruent - rt_congruent

    # --- 4. Results summary ---

    results = {
        "Overall Accuracy": f"{accuracy:.3f}",
        "Overall Error Rate": f"{error_rate:.3f}",
        "Overall Miss Rate": f"{miss_rate:.3f}",
        "Mean RT (Correct trials, ms)": f"{mean_rt_correct:.2f}",
        "Alerting Network Effect (ms)": f"{alerting_effect:.2f}",
        "Orienting Network Effect (ms)": f"{orienting_effect:.2f}",
        "Executive Control Effect (ms)": f"{executive_control_effect:.2f}",
        "--- (Detailed counts) ---": "---",
        "Total trials": total_trials,
        "Valid responses": num_responded,
        "Correct count": num_correct,
        "Error count": num_error,
        "Miss/Excluded count": num_miss
    }

    return results


if __name__ == '__main__':
    # Call function with your filename
    file_path = 'lht.txt'
    """Extract behavioral indicators from a single ANT data file"""
    with open(file_path, 'r', encoding='utf-16') as f:
        content = f.read()
    ant_results = analyze_ant_data(file_path)

    # Print results
    if ant_results:
        print("\n" + "=" * 40)
        print("    ANT Task Behavioral Data Analysis Results")
        print("=" * 40)
        for key, value in ant_results.items():
            print(f"{key:<40}: {value}")
        print("=" * 40)