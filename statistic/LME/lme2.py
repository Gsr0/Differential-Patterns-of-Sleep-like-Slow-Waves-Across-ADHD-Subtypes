import pandas as pd
import numpy as np
import mne
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf # Import statsmodels for linear mixed models

# --- 1. Parameter Settings and Data Loading ---
# Please replace 'your_data_file.xlsx' with the actual path to your Excel file
file_path = 'All slow wave parameter sub electrode.xlsx'
# Please replace 'Sheet1' with the actual worksheet name in your Excel file
sheet_name = 'Sheet1'

# The electrode names list you provided (31 total)
electrode_names = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
    'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5',
    'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8','POz', 'O1','Oz', 'O2'
]

# Parameter column names to be tested
# You can modify or expand this list to analyze all parameters of interest
params_to_test = [
    'maxnegpkamp', 'maxpospkamp', 'mxdnslp', 'mxupslp', 'sw_density', 'mean_duration'
]

# Group definitions
group0_id = 0  # Normal control group
group1_id = 1  # ADHD Type 1
group3_id = 3  # ADHD Type 3

# Define all group comparisons to be performed
# Each tuple: (Group A ID, Group B ID, comparison label for plotting)
comparison_groups = [
    (group1_id, group3_id, f'Group {group1_id} vs {group3_id}'),
    (group0_id, group1_id, f'Group {group0_id} vs {group1_id}'),
    (group0_id, group3_id, f'Group {group0_id} vs {group3_id}')
]

# Permutation test parameters
p_threshold_cluster = 0.025  # p-value threshold for cluster formation (two-sided, i.e., one-sided 0.0125)
p_threshold_monte_carlo = 0.05  # Monte Carlo p-value threshold for cluster significance
n_permutations = 5000  # Number of permutations, recommend at least 1000, 5000 is more stable

# Load data
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print("Excel file loaded successfully.")
    print(f"Data contains {df['SubjectID'].nunique()} subjects.")
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check if the file path is correct.")
    exit()

# --- IMPORTANT: Add 'Age' column for LME demonstration ---
# In actual applications, you should ensure your Excel data contains a real 'Age' column.
# If your data doesn't have an 'Age' column, the following code will generate a random 'Age' column for demonstration.
if 'Age' not in df.columns:
    print("\nWarning: 'Age' column not found. For LME demonstration, randomly generated grouped age data has been added.")
    print("Please ensure your actual data contains an 'Age' column for meaningful LME results.")
    np.random.seed(42) # Set random seed for reproducible results

    # Assign a random age based on group for each subject
    # Simulate potentially different age distributions across groups
    unique_subjects = df['SubjectID'].unique()
    subject_age_map = {}
    for sub_id in unique_subjects:
        # Get the group this subject belongs to
        group_for_sub = df[df['SubjectID'] == sub_id]['Group'].iloc[0]
        if group_for_sub == group0_id: # Normal control group
            subject_age_map[sub_id] = np.random.randint(18, 25) # e.g., 18-24 years
        elif group_for_sub == group1_id: # ADHD Type 1
            subject_age_map[sub_id] = np.random.randint(19, 28) # e.g., 19-27 years
        else: # ADHD Type 3
            subject_age_map[sub_id] = np.random.randint(20, 29) # e.g., 20-28 years
    df['Age'] = df['SubjectID'].map(subject_age_map)
    print("Random 'Age' column added successfully.")

# Ensure 'SubjectID' column is treated as categorical variable
df['SubjectID'] = df['SubjectID'].astype('category')


# --- 2. MNE Preparation: Create electrode info object ---
# Get the actual number and IDs of electrodes used in the data
ch_numbers_in_data = sorted(df['Channel'].unique())
n_channels = len(ch_numbers_in_data)
# Based on the number of electrodes in the data, get corresponding electrode names from your provided list
ch_names = electrode_names[:n_channels]

print(f"\nDetected {n_channels} electrodes in the data.")
print(f"Electrode names to be used: {ch_names}")

# Create MNE info structure containing basic electrode information
info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg') # sfreq can be set arbitrarily as we're not analyzing time series data

# Set electrode positions. MNE will automatically find electrode positions from its built-in standard 10-20 system
montage = mne.channels.read_custom_montage('locations.sfp')
info.set_montage(montage, on_missing='warn') # on_missing='warn' will warn about electrodes without found positions

# --- 3. Define electrode adjacency relationships ---
# This is a key step for cluster-based testing, defining which electrodes are "adjacent"
# MNE can automatically calculate adjacency relationships based on electrode 3D positions
adjacency, adj_ch_names = mne.channels.find_ch_adjacency(info, ch_type='eeg')

# Check if electrode order in adjacency matrix matches the info object
if adj_ch_names != ch_names:
    print("Warning: Electrode order in adjacency matrix doesn't match data, attempting to reindex.")
    # If they don't match, need to reindex the adjacency matrix to ensure correct order
    adj_indices = [adj_ch_names.index(ch) for ch in ch_names]
    adjacency = adjacency[np.ix_(adj_indices, adj_indices)]

print("\nElectrode adjacency matrix created successfully.")


# --- 4. Data Organization: Convert long format data to wide format suitable for permutation testing ---
def prepare_data_for_permutation_test(df_input, param_name, group1, group2):
    """
    Organize data from long format to wide format and separate by group for permutation testing.
    Returns two arrays for the two groups, with shape (n_subjects, n_channels).
    """
    # Filter data for the current parameter and two groups to analyze
    df_param = df_input[df_input['Group'].isin([group1, group2])][['SubjectID', 'Group', 'Channel', param_name]]

    # Pivot/transform data, converting parameter values for each electrode into columns
    df_pivot = df_param.pivot_table(index=['SubjectID', 'Group'], columns='Channel', values=param_name)

    # Ensure pivot table column order matches channel numbers corresponding to ch_names
    # Assume df['Channel'] contains 1-based numbering matching the order of electrode_names
    # If certain channels are missing from data, NaN will be automatically filled
    df_pivot = df_pivot.reindex(columns=range(1, len(ch_names) + 1), fill_value=np.nan)

    # Separate data for two groups and convert to numpy arrays
    data_g1 = df_pivot.loc[df_pivot.index.get_level_values('Group') == group1].values
    data_g2 = df_pivot.loc[df_pivot.index.get_level_values('Group') == group2].values

    return data_g1, data_g2


# --- New Method: Independent samples t-test ---
def run_independent_ttest(data_df, param_name, group1_id, group2_id, ch_names):
    """
    Perform independent samples t-test for each electrode.
    Returns numpy arrays containing t-values and p-values in order matching ch_names.
    """
    # Initialize result arrays filled with NaN
    t_values = np.full(len(ch_names), np.nan)
    p_values = np.full(len(ch_names), np.nan)

    # Filter data for current parameter and two groups
    df_filtered = data_df[data_df['Group'].isin([group1_id, group2_id])][['SubjectID', 'Group', 'Channel', param_name]]

    # Iterate through each electrode
    for i, ch_name in enumerate(ch_names):
        # Assume 'Channel' column numbering starts from 1 and corresponds to ch_names order (i.e., ch_names[0] corresponds to Channel 1)
        ch_num = i + 1

        # Extract data for current electrode and parameter in both groups
        data_g1_ch = df_filtered[(df_filtered['Channel'] == ch_num) & (df_filtered['Group'] == group1_id)][param_name].dropna()
        data_g2_ch = df_filtered[(df_filtered['Channel'] == ch_num) & (df_filtered['Group'] == group2_id)][param_name].dropna()

        # Ensure each group has at least two valid data points for t-test
        if len(data_g1_ch) > 1 and len(data_g2_ch) > 1:
            # Perform independent samples t-test
            # equal_var=True assumes equal variances (Levene's test can be used to check)
            t_stat, p_val = stats.ttest_ind(data_g1_ch, data_g2_ch, equal_var=True)
            t_values[i] = t_stat
            p_values[i] = p_val
        # else:
        #     # If insufficient data, corresponding t-value and p-value remain NaN
        #     print(f"Warning: Electrode {ch_name} (number {ch_num}) has insufficient data in one or both groups for independent samples t-test.")

    return t_values, p_values


# --- New Method: Linear Mixed Model (LME) ---
def run_lme_per_channel(data_df, param_name, group1_id, group2_id, ch_names):
    """
    For each electrode, use age as covariate to perform linear mixed model.
    Returns arrays containing Z-values and P-values for group comparison (group effect) for each electrode, in order matching ch_names.
    Z-values in LME are similar to t-values, used to measure effect significance.
    """
    # Initialize result arrays filled with NaN
    z_values = np.full(len(ch_names), np.nan)
    p_values = np.full(len(ch_names), np.nan)

    # Filter data for current parameter, two groups, age and SubjectID
    df_lme = data_df[data_df['Group'].isin([group1_id, group2_id])][['SubjectID', 'Group', 'Channel', 'Age', param_name]].copy()

    # Ensure 'SubjectID' is categorical variable
    df_lme['SubjectID'] = df_lme['SubjectID'].astype('category')

    # Key modification: Ensure 'Group' column categories only include the two groups in current comparison, and set correct category order
    # This way group1_id becomes the reference level in statsmodels
    df_lme['Group'] = pd.Categorical(df_lme['Group'], categories=[group1_id, group2_id])

    # Iterate through each electrode
    for i, ch_name in enumerate(ch_names):
        ch_num = i + 1 # Assume 'Channel' column numbering starts from 1

        # Extract data for current electrode
        df_channel = df_lme[df_lme['Channel'] == ch_num].copy()

        # Check if current electrode has sufficient valid data in both groups
        # Need at least 2 subjects per group to fit LME
        count_g1 = df_channel[df_channel['Group'] == group1_id].shape[0]
        count_g2 = df_channel[df_channel['Group'] == group2_id].shape[0]

        if count_g1 < 2 or count_g2 < 2:
            # print(f"Warning: Electrode {ch_name} (number {ch_num}) has insufficient data in one or both groups for LME.")
            continue

        try:
            # Define LME model formula:
            # param_name ~ C(Group) + Age
            #   - param_name: dependent variable (slow wave parameter you want to analyze)
            #   - C(Group): treat Group as categorical variable (statsmodels automatically creates dummy variables)
            #   - Age: continuous covariate
            # groups=df_channel['SubjectID']: specify groups for random effects (each SubjectID has a random effect)
            # re_formula='1': indicates adding a random intercept for each SubjectID
            model_formula = f'{param_name} ~ C(Group) + Age'
            model = smf.mixedlm(model_formula, data=df_channel,
                                groups=df_channel['SubjectID'],
                                re_formula='1')

            # Fit model, increase maxiter for convergence help, disp=False suppresses detailed fitting output
            fit_result = model.fit(maxiter=1000, disp=False)

            # Extract group comparison statistics (Z-value and P-value)
            # The term name 'C(Group)[T.{group2_id}]' represents the contrast of group2_id relative to reference group group1_id
            group_term_name = f'C(Group)[T.{group2_id}]'

            if group_term_name in fit_result.pvalues:
                p_values[i] = fit_result.pvalues[group_term_name]
                z_values[i] = fit_result.tvalues[group_term_name] # LME usually uses Z-values, but fit_result.tvalues provides Z-like statistics
            # else:
            #     print(f"Warning: Electrode {ch_name} (number {ch_num}) LME results don't contain group comparison term '{group_term_name}'. May have convergence issues or data anomalies.")

        except Exception as e:
            # Catch errors that may occur during LME fitting (such as non-convergence)
            # print(f"Error: LME model for electrode {ch_name} (number {ch_num}) failed to converge or encountered error: {e}")
            pass # Continue processing next electrode

    return z_values, p_values


# --- 5. Loop through all tests and visualizations ---

print(f"\nStarting statistical testing...")
print(f"P-value threshold for cluster formation: {p_threshold_cluster} (two-sided)")
print(f"Monte Carlo p-value threshold for cluster significance: {p_threshold_monte_carlo}")
print("-" * 50)

# Iterate through each parameter to be tested
for param in params_to_test:
    # Iterate through all defined group comparisons
    for g1_id, g2_id, comparison_label in comparison_groups:
        print(f"\n--- Analyzing parameter: {param} (Comparison: {comparison_label}) ---")

        # --- A. Cluster-based permutation test (MNE) ---
        print("\n=== Cluster-based permutation test (MNE) ===")
        # Prepare data, convert from long format to wide format
        X1, X2 = prepare_data_for_permutation_test(df, param, g1_id, g2_id)

        if X1.shape[0] == 0 or X2.shape[0] == 0:
            print(f"Warning: Parameter '{param}', comparison '{comparison_label}' has no data in one or both groups, skipping this permutation test.")
            continue

        # Calculate degrees of freedom based on number of subjects in current comparison, used to determine t-value threshold
        n_subjects_in_comparison = X1.shape[0] + X2.shape[0]
        df_t_test = n_subjects_in_comparison - 2
        # Prevent negative or too small degrees of freedom causing errors
        if df_t_test <= 0:
            print(f"Warning: Comparison '{comparison_label}' has insufficient subjects for valid t-test (degrees of freedom <= 0). Skipping permutation test.")
            continue

        t_threshold = stats.t.ppf(1.0 - p_threshold_cluster / 2, df=df_t_test)
        print(f"T-value threshold for cluster formation in current comparison ({comparison_label}): {t_threshold:.3f}")

        # Perform cluster-based permutation test
        t_obs_mne, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
            [X1, X2],
            n_permutations=n_permutations,
            threshold=t_threshold,
            adjacency=adjacency,
            tail=0, # Two-sided test
            n_jobs=-1 # Use all CPU cores for parallel computation
        )

        # Find significant clusters
        good_clusters_indices = np.where(cluster_p_values < p_threshold_monte_carlo)[0]

        print(f"Results: Found {len(clusters)} clusters in total.")
        if len(good_clusters_indices) > 0:
            print(f"Found {len(good_clusters_indices)} significant clusters (p < {p_threshold_monte_carlo}).")
            for i, cluster_idx in enumerate(good_clusters_indices):
                cluster_p = cluster_p_values[cluster_idx]
                # clusters[cluster_idx] is a boolean mask marking electrodes in this cluster
                ch_inds = np.where(clusters[cluster_idx])[0]
                cluster_chans = [ch_names[i] for i in ch_inds]
                print(f"  - Significant cluster {i + 1}: p = {cluster_p:.4f}, includes electrodes: {cluster_chans}")
        else:
            print("No significant clusters found.")

        # Visualize MNE results
        fig, ax = plt.subplots(figsize=(6, 5))
        title_mne = f'Permutation Cluster Test (t-values): {param}\n({comparison_label})'
        # Prepare mask for marking significant electrodes on the plot
        sig_chans_mask_mne = np.zeros(n_channels, dtype=bool)
        if len(good_clusters_indices) > 0:
            # Set electrode positions in all significant clusters to True in the mask
            for idx in good_clusters_indices:
                sig_chans_mask_mne[clusters[idx]] = True

        im, cn = mne.viz.plot_topomap(
            data=t_obs_mne, # Plot original t-values for each electrode
            pos=info,
            axes=ax,
            show=False,
            cmap='RdBu_r', # Red-blue colormap with 0 in the middle
            mask=sig_chans_mask_mne, # Mark significant electrodes
            mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=4)
        )
        ax.set_title(title_mne, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('t-value')
        plt.show()

        # --- B. Independent samples t-test ---
        print("\n=== Independent samples t-test ===")
        # Call newly defined function to perform independent t-test
        t_obs_ttest, p_values_ttest = run_independent_ttest(df, param, g1_id, g2_id, ch_names)

        # Find significant electrodes (e.g., p < 0.05)
        sig_chans_mask_ttest = p_values_ttest < 0.05

        print(f"Found {np.sum(sig_chans_mask_ttest)} electrodes significant in independent t-test (p < 0.05).")
        if np.sum(sig_chans_mask_ttest) > 0:
            sig_ttest_chans = [ch_names[i] for i, is_sig in enumerate(sig_chans_mask_ttest) if is_sig]
            print(f"  Significant electrodes: {sig_ttest_chans}")
        else:
            print("No significant electrodes found.")

        # Visualize independent t-test results
        fig, ax = plt.subplots(figsize=(6, 5))
        title_ttest = f'Independent t-test (t-values): {param}\n({comparison_label})'
        im, cn = mne.viz.plot_topomap(
            data=t_obs_ttest, # Plot t-values
            pos=info,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            mask=sig_chans_mask_ttest, # Mark significant electrodes
            mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=4)
        )
        ax.set_title(title_ttest, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('t-value')
        plt.show()

        # --- C. Linear Mixed Model (LME) ---
        print("\n=== Linear Mixed Model (LME) - with Age covariate ===")
        # Call newly defined function to perform LME
        z_obs_lme, p_values_lme = run_lme_per_channel(df, param, g1_id, g2_id, ch_names)

        # Find significant electrodes (e.g., p < 0.05)
        sig_chans_mask_lme = p_values_lme < 0.05

        print(f"Found {np.sum(sig_chans_mask_lme)} electrodes significant in LME (p < 0.05).")
        if np.sum(sig_chans_mask_lme) > 0:
            sig_lme_chans = [ch_names[i] for i, is_sig in enumerate(sig_chans_mask_lme) if is_sig]
            print(f"  Significant electrodes: {sig_lme_chans}")
        else:
            print("No significant electrodes found.")

        # Visualize LME results
        fig, ax = plt.subplots(figsize=(6, 5))
        title_lme = f'LME (z-values, with Age): {param}\n({comparison_label})'
        im, cn = mne.viz.plot_topomap(
            data=z_obs_lme, # Plot LME Z-values
            pos=info,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            mask=sig_chans_mask_lme, # Mark significant electrodes
            mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=4)
        )
        ax.set_title(title_lme, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('z-value')
        plt.show()
        print("-" * 50)

print("\nAll analyses completed.")