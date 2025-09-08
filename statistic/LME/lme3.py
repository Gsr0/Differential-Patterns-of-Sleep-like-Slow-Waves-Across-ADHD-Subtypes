import pandas as pd
import numpy as np
import mne
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings('ignore')

# --- 1. Parameter Settings and Data Loading ---
# Please replace 'your_data_file.xlsx' with the actual path to your Excel file
file_path = 'All slow wave parameter sub electrode.xlsx'
# Please replace 'Sheet1' with the actual worksheet name in your Excel file
sheet_name = 'Sheet1'

# The electrode names list you provided (31 total, but actually using 30)
electrode_names = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
    'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5',
    'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8','POz', 'O1','Oz', 'O2'
]

# Parameter column names to be tested
params_to_test = [
    'maxnegpkamp', 'maxpospkamp', 'mxdnslp', 'mxupslp', 'sw_density', 'mean_duration'
]

# Group definitions
group1_id = 1  # ADHD Type 1
group3_id = 3  # ADHD Type 3

# Permutation test parameters
p_threshold_cluster = 0.025  # p-value threshold for cluster formation
p_threshold_monte_carlo = 0.05  # Monte Carlo p-value threshold for cluster significance
n_permutations = 5000  # Number of permutations

# Load data
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print("Excel file loaded successfully.")
    print(f"Data contains {df['SubjectID'].nunique()} subjects.")
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check if the file path is correct.")
    exit()

# --- 2. MNE Preparation: Create electrode info object ---
ch_numbers_in_data = sorted(df['Channel'].unique())
n_channels = len(ch_numbers_in_data)
ch_names = electrode_names[:n_channels]

print(f"\nDetected {n_channels} electrodes in the data.")
print(f"Electrode names to be used: {ch_names}")

# Create MNE info structure
info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')

# Set electrode positions
montage = mne.channels.read_custom_montage('locations.sfp')
info.set_montage(montage, on_missing='warn')

# --- 3. Define electrode adjacency relationships ---
adjacency, adj_ch_names = mne.channels.find_ch_adjacency(info, ch_type='eeg')

if adj_ch_names != ch_names:
    print("Warning: Electrode order in adjacency matrix doesn't match data, attempting to reindex.")
    adj_indices = [adj_ch_names.index(ch) for ch in ch_names]
    adjacency = adjacency[np.ix_(adj_indices, adj_indices)]

print("\nElectrode adjacency matrix created successfully.")


# --- 4. LME Analysis Function ---
def run_lme_analysis(df, param_name, group1, group2):
    """
    Perform analysis using Linear Mixed Effects model with age as covariate
    Returns t-values and p-values for each electrode
    """
    # Filter data for the required groups and parameter
    df_analysis = df[df['Group'].isin([group1, group2])].copy()

    # Check if Age column exists
    if 'Age' not in df_analysis.columns:
        print(f"Warning: 'Age' column not found in data, will not use age as covariate")
        use_age = False
    else:
        use_age = True

    t_values = []
    p_values = []

    # Perform LME analysis for each electrode separately
    for channel in sorted(df_analysis['Channel'].unique()):
        try:
            # Extract data for current electrode
            channel_data = df_analysis[df_analysis['Channel'] == channel].copy()

            # Check data integrity
            if channel_data[param_name].isna().any():
                print(f"Warning: Parameter {param_name} has missing values at electrode {channel}, skipping")
                t_values.append(0)
                p_values.append(1)
                continue

            # Prepare LME model formula
            if use_age:
                formula = f"{param_name} ~ Group + Age"
            else:
                formula = f"{param_name} ~ Group"

            # Fit LME model (using SubjectID as random effect)
            # Note: Since each subject has only one observation per electrode, this is essentially ordinary linear regression
            # But we maintain the LME framework for future extension
            try:
                model = mixedlm(formula, channel_data, groups=channel_data["SubjectID"])
                result = model.fit(method='lbfgs')

                # Extract t-value and p-value for group effect
                group_coef_name = 'Group'
                if group_coef_name in result.params.index:
                    t_val = result.tvalues[group_coef_name]
                    p_val = result.pvalues[group_coef_name]
                else:
                    # If Group coefficient is not found, might be encoding issue, try other possible names
                    group_params = [param for param in result.params.index if 'Group' in str(param)]
                    if group_params:
                        t_val = result.tvalues[group_params[0]]
                        p_val = result.pvalues[group_params[0]]
                    else:
                        t_val = 0
                        p_val = 1

                t_values.append(t_val)
                p_values.append(p_val)

            except Exception as e:
                print(f"LME model fitting failed for electrode {channel}: {str(e)}")
                # If LME fails, fallback to ordinary t-test
                group1_data = channel_data[channel_data['Group'] == group1][param_name]
                group2_data = channel_data[channel_data['Group'] == group2][param_name]

                if len(group1_data) > 0 and len(group2_data) > 0:
                    t_stat, p_val = stats.ttest_ind(group1_data, group2_data)
                    t_values.append(t_stat)
                    p_values.append(p_val)
                else:
                    t_values.append(0)
                    p_values.append(1)

        except Exception as e:
            print(f"Error analyzing electrode {channel}: {str(e)}")
            t_values.append(0)
            p_values.append(1)

    return np.array(t_values), np.array(p_values)


# --- 5. Cluster Test Function Based on LME Results ---
def lme_cluster_permutation_test(df, param_name, group1, group2, adjacency,
                                 n_permutations=1000, p_threshold=0.025):
    """
    Perform cluster permutation test based on LME analysis results
    """
    # Get original LME analysis results
    original_t, original_p = run_lme_analysis(df, param_name, group1, group2)

    # Create clusters (based on p-value threshold)
    significant_mask = original_p < p_threshold

    # If no significant electrodes, return empty results
    if not np.any(significant_mask):
        return original_t, [], [], []

    # Use MNE's cluster detection algorithm
    # First convert t-values to t-threshold corresponding to p-value threshold
    df_residual = len(df['SubjectID'].unique()) - 2  # Degrees of freedom estimate
    t_threshold = stats.t.ppf(1.0 - p_threshold / 2, df=df_residual)

    # Create pseudo data for permutation test
    df_perm = df[df['Group'].isin([group1, group2])].copy()

    # Store permutation results
    cluster_stats = []

    # Detect clusters in original data
    clusters = []
    cluster_t_sums = []

    # Simplified cluster detection: find connected significant regions based on adjacency matrix
    visited = np.zeros(len(original_t), dtype=bool)

    for i in range(len(original_t)):
        if significant_mask[i] and not visited[i]:
            # Start a new cluster
            cluster = []
            stack = [i]

            while stack:
                current = stack.pop()
                if visited[current]:
                    continue

                visited[current] = True
                cluster.append(current)

                # Check adjacent electrodes
                for j in range(len(original_t)):
                    if (not visited[j] and significant_mask[j] and
                            adjacency[current, j]):
                        stack.append(j)

            if cluster:
                clusters.append(cluster)
                cluster_t_sums.append(np.sum(np.abs(original_t[cluster])))

    # Permutation test
    null_distribution = []

    for perm in range(n_permutations):
        # Randomly permute group labels
        df_shuffled = df_perm.copy()
        subjects = df_shuffled['SubjectID'].unique()
        np.random.shuffle(subjects)

        # Create permuted group label mapping
        n_group1 = len(df_perm[df_perm['Group'] == group1]['SubjectID'].unique())
        group1_subjects = subjects[:n_group1]
        group2_subjects = subjects[n_group1:]

        # Apply permutation
        df_shuffled.loc[df_shuffled['SubjectID'].isin(group1_subjects), 'Group'] = group1
        df_shuffled.loc[df_shuffled['SubjectID'].isin(group2_subjects), 'Group'] = group2

        # Calculate statistics after permutation
        perm_t, perm_p = run_lme_analysis(df_shuffled, param_name, group1, group2)
        perm_significant = perm_p < p_threshold

        # Detect maximum cluster statistic in permuted data
        max_cluster_stat = 0
        visited_perm = np.zeros(len(perm_t), dtype=bool)

        for i in range(len(perm_t)):
            if perm_significant[i] and not visited_perm[i]:
                cluster_perm = []
                stack = [i]

                while stack:
                    current = stack.pop()
                    if visited_perm[current]:
                        continue

                    visited_perm[current] = True
                    cluster_perm.append(current)

                    for j in range(len(perm_t)):
                        if (not visited_perm[j] and perm_significant[j] and
                                adjacency[current, j]):
                            stack.append(j)

                if cluster_perm:
                    cluster_stat = np.sum(np.abs(perm_t[cluster_perm]))
                    max_cluster_stat = max(max_cluster_stat, cluster_stat)

        null_distribution.append(max_cluster_stat)

    # Calculate cluster p-values
    cluster_p_values = []
    for cluster_stat in cluster_t_sums:
        p_val = np.mean([null_stat >= cluster_stat for null_stat in null_distribution])
        cluster_p_values.append(p_val)

    return original_t, clusters, cluster_p_values, cluster_t_sums


# --- 6. Data Preparation Function (keeping original t-test method as backup) ---
def prepare_data_for_test(df, param_name, group1, group2):
    """
    Reshape data from long format to wide format for traditional permutation test
    """
    df_param = df[df['Group'].isin([group1, group2])][['SubjectID', 'Group', 'Channel', param_name]]
    df_pivot = df_param.pivot_table(index=['SubjectID', 'Group'], columns='Channel', values=param_name)

    data_g1 = df_pivot.loc[df_pivot.index.get_level_values('Group') == group1].values
    data_g3 = df_pivot.loc[df_pivot.index.get_level_values('Group') == group2].values

    return data_g1, data_g3


# --- 7. Main Analysis Loop ---
print(f"\nStarting LME statistical testing...")
print(f"P-value threshold for cluster formation: {p_threshold_cluster}")
print(f"Monte Carlo p-value threshold for cluster significance: {p_threshold_monte_carlo}")
print("-" * 50)

# Choose analysis method
USE_LME = True  # Set to True for LME method, False for original t-test method

for param in params_to_test:
    print(f"Analyzing parameter: {param}")

    if USE_LME:
        # Use LME method
        print("Using Linear Mixed Effects (LME) model for analysis...")

        # Perform LME cluster permutation test
        t_obs, clusters, cluster_p_values, cluster_stats = lme_cluster_permutation_test(
            df, param, group1_id, group3_id, adjacency, n_permutations, p_threshold_cluster
        )

        # Find significant clusters
        good_clusters_indices = np.where(np.array(cluster_p_values) < p_threshold_monte_carlo)[0]

    else:
        # Use original t-test method
        print("Using traditional t-test method...")

        # Prepare data
        X1, X2 = prepare_data_for_test(df, param, group1_id, group3_id)

        if X1.shape[0] == 0 or X2.shape[0] == 0:
            print(f"Warning: One or both groups have no data for parameter '{param}', skipping this parameter.")
            continue

        # Calculate t-threshold
        t_threshold = stats.t.ppf(1.0 - p_threshold_cluster / 2, df=df['SubjectID'].nunique() - 2)

        # Perform traditional cluster-based permutation test
        t_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
            [X1, X2],
            n_permutations=n_permutations,
            threshold=t_threshold,
            adjacency=adjacency,
            tail=0,
            n_jobs=-1
        )

        good_clusters_indices = np.where(cluster_p_values < p_threshold_monte_carlo)[0]

    # Output results
    print(f"Results: Found {len(clusters)} clusters in total.")
    if len(good_clusters_indices) > 0:
        print(f"Found {len(good_clusters_indices)} significant clusters (p < {p_threshold_monte_carlo}).")
        for i, cluster_idx in enumerate(good_clusters_indices):
            if USE_LME:
                cluster_p = cluster_p_values[cluster_idx]
                ch_inds = clusters[cluster_idx]
            else:
                cluster_p = cluster_p_values[cluster_idx]
                ch_inds = np.where(clusters[cluster_idx])[0]

            cluster_chans = [ch_names[i] for i in ch_inds]
            print(f"  - Significant cluster {i + 1}: p = {cluster_p:.4f}, includes electrodes: {cluster_chans}")
    else:
        print("No significant clusters found.")

    # --- 8. Visualization: Plot scalp topography ---
    sig_chans_mask = np.zeros(n_channels, dtype=bool)
    if len(good_clusters_indices) > 0:
        for idx in good_clusters_indices:
            if USE_LME:
                sig_chans_mask[clusters[idx]] = True
            else:
                sig_chans_mask[clusters[idx]] = True

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    if USE_LME:
        title = f'LME t-values: {param} (Group {group1_id} vs {group3_id})\nwith Age as covariate'
    else:
        title = f'T-test t-values: {param} (Group {group1_id} vs {group3_id})'

    im, cn = mne.viz.plot_topomap(
        data=t_obs,
        pos=info,
        axes=ax,
        show=False,
        cmap='RdBu_r',
        mask=sig_chans_mask,
        mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k',
                         linewidth=0, markersize=6)
    )

    ax.set_title(title, fontweight='bold')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('t-value')

    plt.tight_layout()
    plt.show()

    # If using LME, can also output more detailed statistical information
    if USE_LME:
        # Perform multiple comparisons correction
        _, p_corrected, _, _ = multipletests(
            run_lme_analysis(df, param, group1_id, group3_id)[1],
            method='fdr_bh'
        )

        print(f"Significant electrodes after multiple comparisons correction (FDR < 0.05):")
        sig_electrodes_corrected = np.where(p_corrected < 0.05)[0]
        if len(sig_electrodes_corrected) > 0:
            for idx in sig_electrodes_corrected:
                print(f"  - {ch_names[idx]}: t = {t_obs[idx]:.3f}, p_corrected = {p_corrected[idx]:.4f}")
        else:
            print("  No significant electrodes")

    print("-" * 50)

print("\nAnalysis completed!")


# --- 9. Supplementary: Save results to Excel ---
def save_results_to_excel(df, params_to_test, group1_id, group3_id, filename='LME_analysis_results.xlsx'):
    """
    Save LME analysis results to Excel file
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for param in params_to_test:
            t_values, p_values = run_lme_analysis(df, param, group1_id, group3_id)

            # Create results DataFrame
            results_df = pd.DataFrame({
                'Channel': range(1, len(t_values) + 1),
                'Electrode': ch_names[:len(t_values)],
                't_value': t_values,
                'p_value': p_values
            })

            # Add multiple comparisons correction
            _, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
            results_df['p_corrected_FDR'] = p_corrected
            results_df['significant_uncorrected'] = p_values < 0.05
            results_df['significant_FDR'] = p_corrected < 0.05

            # Save to different sheets
            results_df.to_excel(writer, sheet_name=param, index=False)

    print(f"Results saved to {filename}")

# If you need to save results, uncomment the line below
# save_results_to_excel(df, params_to_test, group1_id, group3_id)