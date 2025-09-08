import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Set pandas display options to show all columns and full content
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # No display width limit
pd.set_option('display.max_colwidth', None) # No column width limit
pd.set_option('display.expand_frame_repr', False)  # No line breaks

# --- 1. Data Loading and Preparation ---

try:
    # Load your Excel file
    df = pd.read_excel('slow wave parameter in cluster.xlsx')
except FileNotFoundError:
    print("Error: Please ensure the file named 'slow wave parameter in cluster.xlsx' is in the same folder as this script.")
    exit()

# For convenience, we assume the following column names, please modify according to your actual column names
# Assumption: 'Group' is the grouping variable, 'Age' is age
# 'Executive_Control' is the executive control score (dependent variable Y)
# 'maxnegpkamp' is your most important slow wave parameter (independent variable X)
# Please be sure to replace the 'column_name' below with your real column names

# BEHAVIOR_Y = 'Accuracy'
# BEHAVIOR_Y = 'Response Time'  # Behavioral - Reaction Time (RT)
# BEHAVIOR_Y = 'Alerting'  # ANT - Alerting network
BEHAVIOR_Y = 'Orienting'  # ANT - Orienting network
# BEHAVIOR_Y = 'Executive Control'  # ANT - Executive control network

# BRAIN_X = 'maxnegpkamp'
# BRAIN_X = 'maxnegpkamp_Fp1'  # Executive Control+++
# BRAIN_X = 'maxnegpkamp_3'
# BRAIN_X = 'maxnegpkamp_5'
# BRAIN_X = 'maxnegpkamp_F4'  # Orienting
# BRAIN_X = 'maxnegpkamp_9' # Orienting
# BRAIN_X = 'maxnegpkamp_10'
# BRAIN_X = 'maxnegpkamp_12'
# BRAIN_X = 'maxnegpkamp_cluster1' # Orienting
# BRAIN_X = 'maxnegpkamp_cluster2'
# BRAIN_X = 'maxnegpkamp_cluster' # Orienting
# BRAIN_X = 'maxnegpkamp/mxdnslp_1'
# BRAIN_X = 'mxdnslp_1'
# BRAIN_X = 'mxdnslp_3' # Orienting
# BRAIN_X = 'mxdnslp_4'
# BRAIN_X = 'mxdnslp_5'
# BRAIN_X = 'mxdnslp_7'
# BRAIN_X = 'mxdnslp_8'
# BRAIN_X = 'mxdnslp_9'
# BRAIN_X = 'mxdnslp_12'  # ACC
# BRAIN_X = 'mxdnslp_14'
# BRAIN_X = 'mxdnslp_18'
# BRAIN_X = 'mxdnslp_cluster1' # Orienting
# BRAIN_X = 'mxdnslp_cluster2' # Orienting
# BRAIN_X = 'mxdnslp_cluster' # Orienting
# BRAIN_X = 'mxupslp_cluster'  # Executive Control
# BRAIN_X = 'mxupslp_1'  # Executive Control
# BRAIN_X = 'mxupslp_3'
# BRAIN_X = 'mxupslp_7'
# BRAIN_X = 'mxupslp_5'
BRAIN_X = 'maxpospkamp' # Executive Control  # Orienting
# BRAIN_X = 'mxdnslp'  # Slow wave parameter - Maximum downward slope
# BRAIN_X = 'mxupslp'  # Slow wave parameter - Maximum upward slope
# BRAIN_X = 'sw_density' # Slow wave parameter - Slow wave density
# BRAIN_X = 'mean_duration'  # Slow wave parameter - Mean duration


# --- 2. Filter Data and Create Moderator and Centered Variables ---

# Filter data for ADHD-I (value 1) and ADHD-C (value 3)
df_subset = df[df['Group'].isin([1, 3])].copy()

# Create moderator variable W: Group_ADHD_C
# ADHD-I group as 0, ADHD-C group as 1. This makes results easier to interpret
df_subset['Group_ADHD_C'] = df_subset['Group'].apply(lambda x: 1 if x == 3 else 0)

# Center continuous independent variables and covariates (subtract mean)
# This reduces multicollinearity and makes main effects easier to interpret
df_subset[f'{BRAIN_X}_centered'] = df_subset[BRAIN_X] - df_subset[BRAIN_X].mean()
df_subset['Age_centered'] = df_subset['Age'] - df_subset['Age'].mean()


# --- 3. Run Moderation Analysis ---

print("--- Moderation Effect Analysis ---")
# Use pingouin.linear_regression to build a model with interaction term
# Formula: Y ~ X + W + X*W + Covariate
# Y = BEHAVIOR_Y, X = BRAIN_X_centered, W = Group_ADHD_C
model = pg.linear_regression(
    X=df_subset[[f'{BRAIN_X}_centered', 'Group_ADHD_C', 'Age_centered']],
    y=df_subset[BEHAVIOR_Y],
    add_intercept=True
)

# Manually add interaction term
interaction_term = df_subset[f'{BRAIN_X}_centered'] * df_subset['Group_ADHD_C']
X_with_interaction = sm.add_constant(pd.concat([
    df_subset[[f'{BRAIN_X}_centered', 'Group_ADHD_C', 'Age_centered']],
    interaction_term.rename('Interaction')
], axis=1))

model_with_interaction = sm.OLS(df_subset[BEHAVIOR_Y], X_with_interaction).fit()

print("Model Results (Interaction term is 'Interaction'):")
print(model_with_interaction.summary())
print("\n" + "="*50 + "\n")

# Key interpretation: Check the P>|t| (p-value) for the 'Interaction' row in the results above.
# If this p-value is less than 0.05, it indicates a significant moderation effect!


# --- 4. Simple Slopes Analysis (Post-Hoc Analysis) ---
# Since the moderation effect is significant, we need to understand the nature of the relationship
print("--- Simple Slopes Analysis (Post-Hoc Test) ---")

# Analyze ADHD-I group (Group_ADHD_C = 0)
df_i = df_subset[df_subset['Group_ADHD_C'] == 0]
slope_i = pg.linear_regression(X=df_i[[f'{BRAIN_X}_centered', 'Age_centered']], y=df_i[BEHAVIOR_Y])
print("ADHD-I Group: Effect of Slow-wave on Behavior")
print(slope_i.round(3))
print("\n")


# Analyze ADHD-C group (Group_ADHD_C = 1)
df_c = df_subset[df_subset['Group_ADHD_C'] == 1]
slope_c = pg.linear_regression(X=df_c[[f'{BRAIN_X}_centered', 'Age_centered']], y=df_c[BEHAVIOR_Y])
print("ADHD-C Group: Effect of Slow-wave on Behavior")
print(slope_c.round(3))
print("\n" + "="*50 + "\n")


# --- 5. Visualization ---
print("--- Generating Moderation Effect Visualization... ---")

# Use seaborn's lmplot function and assign the returned FacetGrid object to variable 'g'
# 'g' now contains all information about the plot, including the legend
g = sns.lmplot(
    data=df_subset,
    x=BRAIN_X,
    y=BEHAVIOR_Y,
    hue='Group_ADHD_C',    # Use this 0/1 variable to differentiate colors
    ci=None,               # Don't show confidence intervals for clearer visualization
    palette=['#2b6a99', '#f16c23'], # Specify different colors for ADHD-I and ADHD-C
    height=6,
    aspect=1.15
)

# Use seaborn's new function sns.move_legend() to move the legend to the upper right
# This is a more robust method than manually creating a new legend
sns.move_legend(g, "upper right", fontsize=13)

# Now we can modify the legend that has been moved to the upper right
# 1. Set the legend title ("Group_ADHD_C") to an empty string to hide it
g.legend.set_title('')

# 2. Define new labels
new_labels = ['ADHD-I', 'ADHD-C']

# 3. Iterate through the text objects in the legend and set new labels
#    This replaces the original '0' and '1'
for t, l in zip(g.legend.texts, new_labels):
    t.set_text(l)

# Set X and Y axis labels
plt.xlabel(f'Slow-wave Parameter ({BRAIN_X})', fontsize=19)
plt.ylabel(f'ANT network ({BEHAVIOR_Y})', fontsize=19)

# Display the final plot
plt.show()