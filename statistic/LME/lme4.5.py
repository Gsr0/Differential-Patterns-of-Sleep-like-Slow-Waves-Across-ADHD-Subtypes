import pandas as pd
import numpy as np
import mne
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm
import warnings

warnings.filterwarnings("ignore")

# --- 1. Parameter Settings and Data Loading ---
file_path = 'All slow wave parameter sub electrode.xlsx'
sheet_name = 'Sheet1'

# Electrode name list
electrode_names = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
    'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5',
    'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'
]

# Parameter column names to be tested
params_to_test = [
    'maxnegpkamp', 'mxdnslp'
]

# Group definitions
group1_id = 1  # ADHD Type 1
group3_id = 3  # ADHD Type 3

# Define all comparison combinations
comparisons = [
    (group1_id, group3_id, "ADHD Type 1 vs ADHD Type 3")
]

# Permutation test parameters
p_threshold_cluster = 0.04  # p-value threshold for cluster formation
p_threshold_monte_carlo = 0.05  # Monte Carlo p-value threshold for cluster significance
n_permutations = 5000  # Number of permutations

# Load data
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print("Excel file loaded successfully.")
    print(f"Data contains {df['SubjectID'].nunique()} subjects.")

    # Print subject count for each group
    group_counts = df.groupby('Group')['SubjectID'].nunique()
    print("Subject count per group:")
    for group_id in [group1_id, group3_id]:
        if group_id in group_counts.index:
            print(f"  Group {group_id}: {group_counts[group_id]} subjects")
        else:
            print(f"  Group {group_id}: 0 subjects (Warning: No data for this group)")

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check if the file path is correct.")
    exit()

# --- 2. MNE Preparation: Create electrode info object ---
ch_numbers_in_data = sorted(df['Channel'].unique())
n_channels = len(ch_numbers_in_data)
ch_names = electrode_names[:n_channels]

print(f"\nDetected {n_channels} electrodes in data.")
print(f"Electrode names to be used: {ch_names}")

# Create MNE info structure
info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')

# Set electrode positions
montage = mne.channels.read_custom_montage('locations.sfp')
info.set_montage(montage, on_missing='warn')

# --- 3. Define electrode adjacency relationships ---
adjacency, adj_ch_names = mne.channels.find_ch_adjacency(info, ch_type='eeg')

if adj_ch_names != ch_names:
    print("Warning: Adjacency matrix electrode order doesn't match data, attempting to reindex.")
    adj_indices = [adj_ch_names.index(ch) for ch in ch_names]
    adjacency = adjacency[np.ix_(adj_indices, adj_indices)]

print("\nElectrode adjacency matrix created successfully.")


# --- 4. LME Model Function Definition ---
def fit_lme_model(data, param_name, group1_val, group2_val):
    """
    Fit linear mixed effects model for single electrode data

    Parameters:
    -----------
    data : DataFrame
        DataFrame containing SubjectID, Group, Age and parameter values
    param_name : str
        Parameter name to analyze
    group1_val : int
        First group value (as reference group)
    group2_val : int
        Second group value

    Returns:
    --------
    t_stat : float
        t-statistic for Group effect
    p_value : float
        p-value for Group effect
    """
    try:
        # Convert Group to categorical variable with group1_val as reference
        data = data.copy()
        data['Group'] = data['Group'].map({group1_val: 0, group2_val: 1})

        # Standardize Age for numerical stability
        data['Age_std'] = (data['Age'] - data['Age'].mean()) / data['Age'].std()

        # Fit linear mixed effects model
        # Using simple random intercept model with SubjectID as random effect
        formula = f"{param_name} ~ Group + Age_std"
        model = mixedlm(formula, data, groups=data['SubjectID'],
                        missing='drop')
        result = model.fit(method='lbfgs', maxiter=1000)

        # Extract t-statistic and p-value for Group effect
        group_idx = result.params.index.get_loc('Group')
        t_stat = result.tvalues[group_idx]
        p_value = result.pvalues[group_idx]

        return t_stat, p_value

    except Exception as e:
        print(f"Error fitting LME model: {e}")
        return np.nan, np.nan


def prepare_data_for_lme(df, param_name, group1, group2):
    """
    Prepare data for LME model

    Returns:
    --------
    data_list : list
        List of DataFrames for each electrode
    """
    # Filter data for current parameter and two groups
    df_param = df[df['Group'].isin([group1, group2])][
        ['SubjectID', 'Group', 'Age', 'Channel', param_name]
    ].dropna()

    data_list = []
    for ch in range(1, n_channels + 1):
        ch_data = df_param[df_param['Channel'] == ch].copy()
        if len(ch_data) > 0:
            data_list.append(ch_data)
        else:
            data_list.append(None)

    return data_list


# --- 5. Custom LME Permutation Test Function ---
def lme_permutation_cluster_test(data_list, param_name, adjacency, group1_val, group2_val,
                                 n_permutations=1000, p_threshold=0.025):
    """
    Perform cluster-based permutation test for LME model

    Parameters:
    -----------
    data_list : list
        List of DataFrames for each electrode
    param_name : str
        Parameter name
    adjacency : sparse matrix
        Electrode adjacency matrix
    group1_val : int
        First group value
    group2_val : int
        Second group value
    n_permutations : int
        Number of permutations
    p_threshold : float
        p-value threshold for cluster formation

    Returns:
    --------
    t_obs : array
        Observed t-statistics
    clusters : list
        List of clusters
    cluster_p_values : array
        Cluster p-values
    """
    n_channels = len(data_list)
    t_obs = np.zeros(n_channels)
    p_obs = np.zeros(n_channels)

    # 1. Calculate observed statistics
    print("Computing observed LME statistics...")
    for ch in range(n_channels):
        if data_list[ch] is not None:
            t_stat, p_val = fit_lme_model(data_list[ch], param_name, group1_val, group2_val)
            t_obs[ch] = t_stat
            p_obs[ch] = p_val
        else:
            t_obs[ch] = 0
            p_obs[ch] = 1

    # 2. Determine significant electrodes based on p-value threshold
    threshold_mask = p_obs < p_threshold

    # 3. Find clusters based on adjacency relationships
    from scipy.sparse.csgraph import connected_components

    # Create adjacency matrix subset containing only significant electrodes
    if np.any(threshold_mask):
        sig_adjacency = adjacency.copy()
        sig_adjacency = sig_adjacency.multiply(
            np.outer(threshold_mask, threshold_mask)
        )

        # Find connected components (clusters)
        n_components, labels = connected_components(
            sig_adjacency, directed=False
        )

        clusters = []
        cluster_stats = []

        for i in range(n_components):
            cluster_mask = (labels == i) & threshold_mask
            if np.sum(cluster_mask) > 0:
                clusters.append(cluster_mask)
                # Cluster statistic is sum of absolute t-values in cluster
                cluster_stat = np.sum(np.abs(t_obs[cluster_mask]))
                cluster_stats.append(cluster_stat)
    else:
        clusters = []
        cluster_stats = []

    if len(clusters) == 0:
        return t_obs, [], np.array([])

    cluster_stats = np.array(cluster_stats)

    # 4. Permutation test
    print(f"Performing {n_permutations} permutation tests...")
    null_cluster_stats = []

    for perm in range(n_permutations):
        if (perm + 1) % 1000 == 0:
            print(f"  Completed {perm + 1}/{n_permutations} permutations")

        # Independently permute Group labels for each electrode
        t_perm = np.zeros(n_channels)
        p_perm = np.zeros(n_channels)

        for ch in range(n_channels):
            if data_list[ch] is not None:
                data_perm = data_list[ch].copy()
                # Permute Group labels
                data_perm['Group'] = np.random.permutation(data_perm['Group'].values)
                t_stat_perm, p_val_perm = fit_lme_model(data_perm, param_name, group1_val, group2_val)
                t_perm[ch] = t_stat_perm
                p_perm[ch] = p_val_perm
            else:
                t_perm[ch] = 0
                p_perm[ch] = 1

        # Find significant electrodes and clusters after permutation
        perm_threshold_mask = p_perm < p_threshold

        if np.any(perm_threshold_mask):
            perm_sig_adjacency = adjacency.copy()
            perm_sig_adjacency = perm_sig_adjacency.multiply(
                np.outer(perm_threshold_mask, perm_threshold_mask)
            )

            perm_n_components, perm_labels = connected_components(
                perm_sig_adjacency, directed=False
            )

            perm_max_cluster_stat = 0
            for i in range(perm_n_components):
                perm_cluster_mask = (perm_labels == i) & perm_threshold_mask
                if np.sum(perm_cluster_mask) > 0:
                    perm_cluster_stat = np.sum(np.abs(t_perm[perm_cluster_mask]))
                    perm_max_cluster_stat = max(perm_max_cluster_stat, perm_cluster_stat)

            null_cluster_stats.append(perm_max_cluster_stat)
        else:
            null_cluster_stats.append(0)

    null_cluster_stats = np.array(null_cluster_stats)

    # 5. Calculate cluster p-values
    cluster_p_values = np.zeros(len(cluster_stats))
    for i, cluster_stat in enumerate(cluster_stats):
        cluster_p_values[i] = np.mean(null_cluster_stats >= cluster_stat)

    return t_obs, clusters, cluster_p_values


# --- 6. Main Analysis Loop ---
print(f"\nStarting LME statistical testing...")
print(f"p-value threshold for cluster formation: {p_threshold_cluster}")
print(f"Monte Carlo p-value threshold for cluster significance: {p_threshold_monte_carlo}")
print(f"Will perform {len(comparisons)} group comparisons")
print("=" * 70)

for param in params_to_test:
    print(f"\nAnalyzing parameter: {param}")
    print("=" * 50)

    for comp_idx, (group1, group2, comp_name) in enumerate(comparisons):
        print(f"\nComparison {comp_idx + 1}/{len(comparisons)}: {comp_name} (Group {group1} vs Group {group2})")

        # Prepare data
        data_list = prepare_data_for_lme(df, param, group1, group2)

        # Check data validity
        valid_channels = sum(1 for data in data_list if data is not None)
        if valid_channels == 0:
            print(f"Warning: Parameter '{param}' has no valid data for this comparison, skipping.")
            continue

        # Check data amount for each group
        total_subjects = set()
        group1_subjects = set()
        group2_subjects = set()

        for data in data_list:
            if data is not None:
                total_subjects.update(data['SubjectID'].unique())
                group1_subjects.update(data[data['Group'] == group1]['SubjectID'].unique())
                group2_subjects.update(data[data['Group'] == group2]['SubjectID'].unique())

        print(f"Data overview: Group {group1} {len(group1_subjects)} subjects, Group {group2} {len(group2_subjects)} subjects")

        if len(group1_subjects) < 5 or len(group2_subjects) < 5:
            print("Warning: Too few subjects in one group (<5), results may be unreliable.")

        # Execute LME permutation test
        t_obs, clusters, cluster_p_values = lme_permutation_cluster_test(
            data_list, param, adjacency, group1, group2,
            n_permutations=n_permutations,
            p_threshold=p_threshold_cluster
        )

        # Find significant clusters
        good_clusters_indices = np.where(cluster_p_values < p_threshold_monte_carlo)[0]

        print(f"Results: Found {len(clusters)} clusters in total.")
        if len(good_clusters_indices) > 0:
            print(f"Found {len(good_clusters_indices)} significant clusters (p < {p_threshold_monte_carlo}).")
            for i, cluster_idx in enumerate(good_clusters_indices):
                cluster_p = cluster_p_values[cluster_idx]
                ch_inds = np.where(clusters[cluster_idx])[0]
                cluster_chans = [ch_names[i] for i in ch_inds]
                print(f"  - Significant cluster {i + 1}: p = {cluster_p:.4f}, electrodes: {cluster_chans}")
        else:
            print("No significant clusters found.")

        # --- 7. Visualization: Plot scalp topography ---
        sig_chans_mask = np.zeros(n_channels, dtype=bool)
        if len(good_clusters_indices) > 0:
            for idx in good_clusters_indices:
                sig_chans_mask[clusters[idx]] = True

        fig, ax = plt.subplots(figsize=(6, 5))
        title = f'LME t-values: {param}\n{comp_name} (Age adjusted)'

        im, cn = mne.viz.plot_topomap(
            data=t_obs,
            pos=info,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            mask=sig_chans_mask,
            mask_params=dict(marker='o', markerfacecolor='k',
                             markeredgecolor='k', linewidth=0, markersize=4)
        )

        ax.set_title(title, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('t-value (LME)')

        plt.tight_layout()
        plt.show()

        print("-" * 30)

    print("=" * 50)

print("\nAll analyses completed!")
print("Summary:")
print(f"- Analyzed {len(params_to_test)} parameters")
print(f"- Performed {len(comparisons)} group comparisons")
print(f"- Each comparison used {n_permutations} permutation tests")