import pandas as pd
import pingouin as pg

# --- 1. Load and Prepare the Data ---

# Load your Excel file.
# Make sure the file is in the same directory as your script, or provide the full path.
try:
    df = pd.read_excel('ant网络雷达图.xlsx')
    # In your screenshot, the file looks like it's named 'image_8b4178.png',
    # but I'm assuming the actual data is in an .xlsx file.
except FileNotFoundError:
    print("Error: Make sure your Excel file is named 'ADHD_data.xlsx' and is in the correct folder.")
    exit()

# Let's rename columns for easier use, based on your Excel sheet.
df.rename(columns={'Executive Control': 'Executive_Control'}, inplace=True)

# --- 2. Set up the Mediation Analysis ---

# We will compare ADHD-I (Group 1) vs ADHD-C (Group 3).
# First, we filter the dataframe to only include these two groups.
df_subset = df[df['Group'].isin([1, 3])].copy()

# Create the independent variable (X). We'll make a new column 'is_ADHD_C'.
# It will be 1 if the subject is ADHD-C, and 0 if they are a Healthy Control.
df_subset['is_ADHD_C'] = (df_subset['Group'] == 3).astype(int)


# --- 3. Run the Mediation Model ---

print("Running Mediation Analysis: (ADHD-C vs ADHD-I)")
print("===========================================")
print("Hypothesis:  maxnegpkamp-> Group -> Executive_Control\n")

# Use pingouin's mediation_analysis function.
# We specify X, M, and Y, and list 'Age' as a covariate.
# The function automatically handles the multiple regressions and bootstrapping.
mediation_results = pg.mediation_analysis(
    data=df_subset,
    x='maxnegpkamp_cluster1',        # Independent variable (0=HC, 1=ADHD-C)
    m='is_ADHD_C',      # Mediator variable (slow wave negative peak amplitude)
    y='Executive_Control', # Dependent variable (behavioral score)
    covar=['Age'],        # Control for the effect of Age
    n_boot=5000,          # Number of bootstrap samples for robust confidence intervals
    seed=42
)

# --- 4. Display and Interpret the Results ---

# The results are presented in a clean table.
pd.set_option('display.float_format', '{:.3f}'.format) # Format for readability
print(mediation_results)

# HOW TO INTERPRET THE TABLE:
# 1. Look at the 'indirect' path (a*b). This is your main result.
#    - 'coef': The size of the indirect effect.
#    - 'pval': The significance of the indirect effect.
#    - 'CI95%': The 95% confidence interval. If this interval DOES NOT include 0, your mediation effect is significant.
#
# 2. Look at the other paths for the full story:
#    - Path c': The 'direct' effect of Group on Behavior after controlling for the mediator.
#    - Path X -> M: Is the Group significantly predicting the slow-wave parameter?
#    - Path M -> Y: Is the slow-wave parameter significantly predicting Behavior?
#    - Path X -> Y: The 'total' effect, before considering the mediator.