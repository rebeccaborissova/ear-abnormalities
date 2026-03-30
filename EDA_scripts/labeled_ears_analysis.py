import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Read the CSV file
df = pd.read_csv('/Users/mansapatel/Downloads/BabyEar4k/diagnosis_result.csv')

# Function to extract the first number from the score string
def get_first_index(score_str):
    """Extract the first number from a string like '7+0+1' -> 7"""
    return int(str(score_str).split('+')[0])

# Extract first index scores for left and right ears
df['L_score'] = df['L01'].apply(get_first_index)
df['R_score'] = df['R01'].apply(get_first_index)

# Define labeled ears
labeled_ears = []

# Both L and R of 0-60
for i in range(0, 61):
    labeled_ears.append(f"{i}_L")
    labeled_ears.append(f"{i}_R")

# Both L and R of 100-150
for i in range(100, 151):
    labeled_ears.append(f"{i}_L")
    labeled_ears.append(f"{i}_R")

# Individual ears
individual_ears = [
    "61_R", "70_L", "71_L", "71_R", "75_R", "82_L", "82_R", "94_L", "97_L", "97_R", 
    "163_L", "169_L", "80_R", "83_L", "88_L", "152_L", "160_L", "166_L", "166_R", 
    "200_L", "231_L", "246_L", "270_L", "284_L", "289_L", "289_R", "298_L"
]
labeled_ears.extend(individual_ears)

# Create a mapping of ear to score
ear_to_score = {}
for idx, row in df.iterrows():
    baby_id = row['baby_id']
    ear_to_score[f"{baby_id}_L"] = row['L_score']
    ear_to_score[f"{baby_id}_R"] = row['R_score']

# Analyze labeled ears
print("=" * 70)
print("LABELED EARS CATEGORY BREAKDOWN")
print("=" * 70)

labeled_scores = []
score_breakdown = {i: [] for i in range(8)}

for ear in labeled_ears:
    if ear in ear_to_score:
        score = ear_to_score[ear]
        labeled_scores.append(score)
        score_breakdown[score].append(ear)

# Print category breakdown
print("\nScore distribution of your labeled ears:")
print("-" * 70)
labeled_counter = Counter(labeled_scores)
total_labeled = len(labeled_scores)

for score in range(8):
    count = labeled_counter.get(score, 0)
    percentage = (count / total_labeled) * 100 if total_labeled > 0 else 0
    print(f"\nScore {score}: {count:4d} ears ({percentage:5.2f}%)")
    if count > 0 and count <= 20:  # Only show ear IDs if there are 20 or fewer
        ears_list = score_breakdown[score][:10]  # Show first 10
        print(f"  Examples: {', '.join(ears_list)}")
        if len(score_breakdown[score]) > 10:
            print(f"  ... and {len(score_breakdown[score]) - 10} more")
    elif count > 20:
        print(f"  Examples: {', '.join(score_breakdown[score][:10])}")
        print(f"  ... and {len(score_breakdown[score]) - 10} more")

print(f"\n{'Total labeled ears:':<30} {total_labeled:4d}")

# Calculate overall dataset statistics
all_scores = list(df['L_score']) + list(df['R_score'])
all_counter = Counter(all_scores)
total_all = len(all_scores)

print("\n" + "=" * 70)
print("COMPARISON: LABELED vs. FULL DATASET")
print("=" * 70)
print(f"\n{'Score':<10} {'Labeled':<15} {'Full Dataset':<15} {'% of Category':<15}")
print("-" * 70)

for score in range(8):
    labeled_count = labeled_counter.get(score, 0)
    all_count = all_counter.get(score, 0)
    pct_labeled = (labeled_count / labeled_scores.__len__()) * 100 if labeled_scores else 0
    pct_all = (all_count / total_all) * 100 if total_all > 0 else 0
    coverage = (labeled_count / all_count) * 100 if all_count > 0 else 0
    print(f"{score:<10} {labeled_count:4d} ({pct_labeled:5.1f}%)   {all_count:4d} ({pct_all:5.1f}%)   {coverage:6.2f}%")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Ear Diagnosis Score Analysis', fontsize=16, fontweight='bold')

# 1. Full dataset distribution
ax1 = axes[0, 0]
scores = range(8)
counts = [all_counter.get(i, 0) for i in scores]
bars1 = ax1.bar(scores, counts, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Score', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Full Dataset - Score Distribution', fontsize=14, fontweight='bold')
ax1.set_xticks(scores)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=10)

# 2. Labeled ears distribution
ax2 = axes[0, 1]
labeled_counts = [labeled_counter.get(i, 0) for i in scores]
bars2 = ax2.bar(scores, labeled_counts, color='coral', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Score', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Labeled Ears - Score Distribution', fontsize=14, fontweight='bold')
ax2.set_xticks(scores)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=10)

# 3. Side-by-side comparison
ax3 = axes[1, 0]
x = np.arange(len(scores))
width = 0.35
bars3a = ax3.bar(x - width/2, counts, width, label='Full Dataset', color='steelblue', alpha=0.7, edgecolor='black')
bars3b = ax3.bar(x + width/2, labeled_counts, width, label='Labeled Ears', color='coral', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Score', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Comparison: Full Dataset vs. Labeled Ears', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(scores)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Coverage percentage
ax4 = axes[1, 1]
coverage_pcts = [(labeled_counter.get(i, 0) / all_counter.get(i, 1)) * 100 for i in scores]
bars4 = ax4.bar(scores, coverage_pcts, color='mediumseagreen', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Score', fontsize=12)
ax4.set_ylabel('Coverage (%)', fontsize=12)
ax4.set_title('Labeled Coverage by Score Category', fontsize=14, fontweight='bold')
ax4.set_xticks(scores)
ax4.grid(axis='y', alpha=0.3)
ax4.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
ax4.legend()

# Add value labels on bars
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('/Users/mansapatel/Downloads/BabyEar4k/labeled_ears_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved to: labeled_ears_visualization.png")
plt.show()

# Additional statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"Total ears in dataset: {total_all}")
print(f"Total labeled ears: {total_labeled}")
print(f"Overall coverage: {(total_labeled/total_all)*100:.2f}%")
print(f"\nMost common score in dataset: {all_counter.most_common(1)[0][0]} ({all_counter.most_common(1)[0][1]} ears)")
print(f"Most common score in labeled: {labeled_counter.most_common(1)[0][0]} ({labeled_counter.most_common(1)[0][1]} ears)")
