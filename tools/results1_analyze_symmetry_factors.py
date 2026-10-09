import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.multiclass import OneVsRestClassifier

# Read the symmetry factors and angles
symmetry_factors = np.load(r'C:\Users\Olive\.spyder-py3\npy_cam_high_hkusz\symmetry_factors.npy', allow_pickle=True)
symmetry_angles = np.load(r'C:\Users\Olive\.spyder-py3\npy_cam_high_hkusz\symmetry_angles.npy', allow_pickle=True)
label_libs = np.load(r'C:\Users\Olive\.spyder-py3\npy_cam_high_hkusz\label_libs_9_128.npy', allow_pickle=True)

# Print basic information
print("Symmetry Factors Shape:", len(symmetry_factors), len(symmetry_angles))
print("Label Libs Shape:", label_libs.shape)

# Convert symmetry factors to a more readable format
symmetry_data = []
for i, data in enumerate(zip(symmetry_factors, symmetry_angles, label_libs[:,-2:])):
    # Each factors is a list of 3 dictionaries (shoulder, elbow, knee)
    factors, angles, labels = data[0], data[1], data[2]
    shoulder_data = factors[0]
    elbow_data = factors[1]
    knee_data = factors[2]
    
    shoulder_angle = angles[0]
    elbow_angle = angles[1]
    knee_angle = angles[2]
    # pdb.set_trace()
    # Get ROM difference metrics for each joint
    shoulder_dist_rom_diff = shoulder_data['shoulder_dist_rom_diff']
    elbow_dist_rom_diff = elbow_data['elbow_dist_rom_diff']
    knee_dist_rom_diff = knee_data['knee_dist_rom_diff']
    
    # Get angle ROM difference metrics
    shoulder_angle_rom_diff = shoulder_angle['shoulder_angle_rom_diff']
    elbow_angle_rom_diff = elbow_angle['elbow_angle_rom_diff']
    knee_angle_rom_diff = knee_angle['knee_angle_rom_diff']
    
    # Determine category based on max label value
    max_label = max(labels)
    if max_label <= 10:
        category = 0
    else:
        category = 1
    
    # Combine all data into one dictionary
    combined_data = {
        'index': i,
        'category': category,
        'shoulder_dist_rom_diff': shoulder_dist_rom_diff,
        'elbow_dist_rom_diff': elbow_dist_rom_diff,
        'knee_dist_rom_diff': knee_dist_rom_diff,
        'shoulder_angle_rom_diff': shoulder_angle_rom_diff,
        'elbow_angle_rom_diff': elbow_angle_rom_diff,
        'knee_angle_rom_diff': knee_angle_rom_diff
    }
    symmetry_data.append(combined_data)

# Convert to DataFrame for better visualization
symmetry_df = pd.DataFrame(symmetry_data)

# Print category distribution
print("\nCategory Distribution:")
category_counts = symmetry_df['category'].value_counts().sort_index()
print(category_counts)

# Define metrics for analysis
dist_rom_diff_factors = ['shoulder_dist_rom_diff', 'elbow_dist_rom_diff', 'knee_dist_rom_diff']
angle_rom_diff_factors = ['shoulder_angle_rom_diff', 'elbow_angle_rom_diff', 'knee_angle_rom_diff']

# Create distribution plots for each category and metric
for category in symmetry_df['category'].unique():
    category_data = symmetry_df[symmetry_df['category'] == category]
    
    # Plot ROM differences
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Distribution of Metrics for Category {category}', fontsize=16)
    
    # Plot distance-based ROM differences
    for idx, factor in enumerate(dist_rom_diff_factors):
        # Create KDE plot
        sns.kdeplot(data=category_data, x=factor, ax=axes[0, idx], fill=True, alpha=0.3)
        
        # Add mean and confidence interval
        mean_val = category_data[factor].mean()
        ci = stats.t.interval(confidence=0.95, df=len(category_data)-1,
                            loc=mean_val,
                            scale=stats.sem(category_data[factor]))
        
        axes[0, idx].axvline(mean_val, color='red', linestyle='--', label='Mean')
        axes[0, idx].axvspan(ci[0], ci[1], color='red', alpha=0.1, label='95% CI')
        
        # Check for bimodality using KDE
        kde = stats.gaussian_kde(category_data[factor])
        x_range = np.linspace(category_data[factor].min(), category_data[factor].max(), 100)
        kde_vals = kde(x_range)
        
        # Find peaks in KDE
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(kde_vals, height=0)
        
        if len(peaks) > 1:  # If bimodal
            # Add vertical lines for peaks
            for peak in x_range[peaks]:
                axes[0, idx].axvline(peak, color='green', linestyle=':', 
                                   label='Subgroup Peak' if peak == x_range[peaks][0] else "")
        
        axes[0, idx].set_title(f'{factor.replace("_", " ").title()}')
        axes[0, idx].set_xlabel('ROM Difference')
        axes[0, idx].set_ylabel('Density')
        axes[0, idx].legend()
    
    # Plot angle-based ROM differences
    for idx, factor in enumerate(angle_rom_diff_factors):
        # Create KDE plot
        sns.kdeplot(data=category_data, x=factor, ax=axes[1, idx], fill=True, alpha=0.3)
        
        # Add mean and confidence interval
        mean_val = category_data[factor].mean()
        ci = stats.t.interval(confidence=0.95, df=len(category_data)-1,
                            loc=mean_val,
                            scale=stats.sem(category_data[factor]))
        
        axes[1, idx].axvline(mean_val, color='red', linestyle='--', label='Mean')
        axes[1, idx].axvspan(ci[0], ci[1], color='red', alpha=0.1, label='95% CI')
        
        # Check for bimodality using KDE
        kde = stats.gaussian_kde(category_data[factor])
        x_range = np.linspace(category_data[factor].min(), category_data[factor].max(), 100)
        kde_vals = kde(x_range)
        
        # Find peaks in KDE
        peaks, _ = find_peaks(kde_vals, height=0)
        
        if len(peaks) > 1:  # If bimodal
            # Add vertical lines for peaks
            for peak in x_range[peaks]:
                axes[1, idx].axvline(peak, color='green', linestyle=':', 
                                   label='Subgroup Peak' if peak == x_range[peaks][0] else "")
        
        axes[1, idx].set_title(f'{factor.replace("_", " ").title()}')
        axes[1, idx].set_xlabel('Angle ROM Difference (degrees)')
        axes[1, idx].set_ylabel('Density')
        axes[1, idx].legend()
    
    plt.tight_layout()
    plt.savefig(f'distribution_category_{category}.png', dpi=300, bbox_inches='tight')
plt.close()

# Group by category and calculate statistics
print("\nCalculating detailed statistics...")

# Create a comprehensive statistics table
stats_columns = [
    'Category', 'Metric', 'Count', 'Mean', 'Std', 'Min', '25%', 'Median', '75%', 'Max',
    'Skewness', 'Kurtosis', 'Effect Size (d)', 'p-value'
]

stats_data = []

for feature in dist_rom_diff_factors + angle_rom_diff_factors:
    for category in symmetry_df['category'].unique():
        category_data = symmetry_df[symmetry_df['category'] == category][feature]
        
        # Calculate basic statistics
        count = len(category_data)
        mean = category_data.mean()
        std = category_data.std()
        min_val = category_data.min()
        q1 = category_data.quantile(0.25)
        median = category_data.median()
        q3 = category_data.quantile(0.75)
        max_val = category_data.max()
        
        # Calculate distribution statistics
        skewness = stats.skew(category_data)
        kurtosis = stats.kurtosis(category_data)
        
        # Calculate effect size (Cohen's d) between categories
        if category == 0:
            other_category = 1
        else:
            other_category = 0
        
        other_data = symmetry_df[symmetry_df['category'] == other_category][feature]
        effect_size = (mean - other_data.mean()) / np.sqrt((std**2 + other_data.std()**2) / 2)
        
        # Calculate p-value using Mann-Whitney U test
        _, p_value = stats.mannwhitneyu(category_data, other_data, alternative='two-sided')
        
        # Add row to stats data
        stats_data.append([
            category, feature, count, mean, std, min_val, q1, median, q3, max_val,
            skewness, kurtosis, effect_size, p_value
        ])

# Create DataFrame and save as CSV
stats_df = pd.DataFrame(stats_data, columns=stats_columns)
stats_df = stats_df.round(4)  # Round all numeric columns to 4 decimal places

# Save to CSV
stats_df.to_csv('symmetry_statistics.csv', index=False)

# Print formatted table
print("\nDetailed Statistics Table (saved as 'symmetry_statistics.csv'):")
print(stats_df.to_string(index=False))

# Create a summary table for quick reference
summary_stats = stats_df.pivot_table(
    index='Metric',
    columns='Category',
    values=['Mean', 'Std', 'Effect Size (d)', 'p-value']
).round(4)

# Save summary to CSV
summary_stats.to_csv('symmetry_summary.csv')

print("\nSummary Statistics (saved as 'symmetry_summary.csv'):")
print(summary_stats)

# Create box plots for ROM differences by category
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('ROM Differences by Category', fontsize=16)

# Plot distance-based ROM differences
for idx, factor in enumerate(dist_rom_diff_factors):
    sns.boxplot(x='category', y=factor, data=symmetry_df, ax=axes[0, idx])
    axes[0, idx].set_title(f'{factor.replace("_", " ").title()}')
    axes[0, idx].set_xlabel('Category')
    axes[0, idx].set_ylabel('ROM Difference')

# Plot angle-based ROM differences
for idx, factor in enumerate(angle_rom_diff_factors):
    sns.boxplot(x='category', y=factor, data=symmetry_df, ax=axes[1, idx])
    axes[1, idx].set_title(f'{factor.replace("_", " ").title()}')
    axes[1, idx].set_xlabel('Category')
    axes[1, idx].set_ylabel('Angle ROM Difference (degrees)')

plt.tight_layout()
plt.savefig('rom_diff_by_category.png', dpi=300, bbox_inches='tight')
plt.close()

# # 10. Box Plots with Statistical Annotations
# for feature in dist_rom_diff_factors + angle_rom_diff_factors:
#     plt.figure(figsize=(10, 6))
#     ax = sns.boxplot(x='category', y=feature, data=symmetry_df)
    
#     # Add statistical annotations
#     for i in range(2):  # Changed from 3 to 2 categories
#         stats_text = f"n={len(symmetry_df[symmetry_df['category']==i])}\n"
#         stats_text += f"mean={symmetry_df[symmetry_df['category']==i][feature].mean():.2f}\n"
#         stats_text += f"std={symmetry_df[symmetry_df['category']==i][feature].std():.2f}"
#         plt.text(i, ax.get_ylim()[0], stats_text, 
#                 horizontalalignment='center', verticalalignment='bottom')
    
#     plt.title(f'{feature} Distribution by Category')
#     plt.savefig(f'boxplot_stats_{feature}.png', dpi=300, bbox_inches='tight')
#     plt.close()
# Advanced Statistical Analysis

print("\n=== Advanced Statistical Analysis ===")
# Prepare data for statistical analysis
X = symmetry_df[dist_rom_diff_factors + angle_rom_diff_factors]
y = symmetry_df['category']

# 1. Correlation Analysis
correlation_matrix = symmetry_df[dist_rom_diff_factors + angle_rom_diff_factors].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of All Metrics')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Shapiro-Wilk Test for Normality
print("\nShapiro-Wilk Test for Normality:")
normality_results = {}
for feature in dist_rom_diff_factors + angle_rom_diff_factors:
    stat, p_value = stats.shapiro(symmetry_df[feature])
    normality_results[feature] = {'statistic': stat, 'p-value': p_value}

normality_df = pd.DataFrame(normality_results).T
print("\nNormality Test Results:")
print(normality_df)

# # 7. Kruskal-Wallis H-test (non-parametric alternative to ANOVA more than 2 groups)
# print("\nKruskal-Wallis H-test Results:")
# kw_results = {}
# for feature in dist_rom_diff_factors + angle_rom_diff_factors:
#     groups = [group[feature].values for name, group in symmetry_df.groupby('category')]
#     h_stat, p_value = stats.kruskal(*groups)
#     kw_results[feature] = {'H-statistic': h_stat, 'p-value': p_value}

# kw_df = pd.DataFrame(kw_results).T
# print("\nKruskal-Wallis Test Results:")
# print(kw_df)

# 8. Mann-Whitney U test (non-parametric t-test)
print("\nMann-Whitney U Test Results:")
mw_results = {}
for feature in dist_rom_diff_factors + angle_rom_diff_factors:
    group1 = symmetry_df[symmetry_df['category'] == 0][feature]
    group2 = symmetry_df[symmetry_df['category'] == 1][feature]
    stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    mw_results[feature] = {'U-statistic': stat, 'p-value': p_value}

mw_df = pd.DataFrame(mw_results).T
print("\nMann-Whitney U Test Results:")
print(mw_df)

# 11. Violin Plots for Distribution Analysis
print("\nGenerating Violin Plots for Distribution Analysis...")

# Create combined violin plots for all metrics
plt.figure(figsize=(20, 12))
plt.suptitle('Distribution of ROM Differences by Category', fontsize=16)

# Plot all metrics in a single figure
for idx, feature in enumerate(dist_rom_diff_factors + angle_rom_diff_factors, 1):
    plt.subplot(2, 3, idx)
    
    # Create violin plot
    ax = sns.violinplot(x='category', y=feature, data=symmetry_df, inner='quartile')
    
    # Calculate means and confidence intervals
    means = symmetry_df.groupby('category')[feature].mean()
    ci_lower = symmetry_df.groupby('category')[feature].apply(
        lambda x: stats.t.interval(confidence=0.95, df=len(x)-1, loc=x.mean(), scale=stats.sem(x))[0])
    ci_upper = symmetry_df.groupby('category')[feature].apply(
        lambda x: stats.t.interval(confidence=0.95, df=len(x)-1, loc=x.mean(), scale=stats.sem(x))[1])
    
    # Add confidence intervals
    for i in range(2):
        ax.errorbar(i, means[i], yerr=[[means[i]-ci_lower[i]], [ci_upper[i]-means[i]]],
                   fmt='o', color='red', capsize=5, label='95% CI' if i==0 else "")
    
    # Add effect size arrow
    effect_size = (means[1] - means[0]) / symmetry_df[feature].std()
    ax.annotate('', xy=(1, means[1]), xytext=(0, means[0]),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text(0.5, (means[0] + means[1])/2, f'd={effect_size:.2f}',
            ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    # Add statistical annotations
    for i in range(2):
        stats_text = f"n={len(symmetry_df[symmetry_df['category']==i])}\n"
        stats_text += f"mean={means[i]:.2f}\n"
        stats_text += f"std={symmetry_df[symmetry_df['category']==i][feature].std():.2f}"
        plt.text(i, ax.get_ylim()[0], stats_text, 
                horizontalalignment='center', verticalalignment='bottom')
    
    plt.title(f'{feature.replace("_", " ").title()}')
    plt.xlabel('Category')
    if idx <= 3:
        plt.ylabel('ROM Difference')
    else:
        plt.ylabel('Angle ROM Difference (degrees)')
    plt.legend()

plt.tight_layout()
plt.savefig('combined_violin_plots.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis complete. All statistical results and visualizations have been saved.")