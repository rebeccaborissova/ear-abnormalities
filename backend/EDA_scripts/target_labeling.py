import pandas as pd
from collections import defaultdict
import random

# Read the CSV file
df = pd.read_csv('/Users/mansapatel/Downloads/BabyEar4k/diagnosis_result.csv')

# Function to extract the first number from the score string
def get_first_index(score_str):
    """Extract the first number from a string like '7+0+1' -> 7"""
    return int(str(score_str).split('+')[0])

# Extract first index scores for left and right ears
df['L_score'] = df['L01'].apply(get_first_index)
df['R_score'] = df['R01'].apply(get_first_index)

# Define already labeled ears
labeled_ears = set()

# Both L and R of 0-60
for i in range(0, 61):
    labeled_ears.add(f"{i}_L")
    labeled_ears.add(f"{i}_R")

# Both L and R of 100-150
for i in range(100, 151):
    labeled_ears.add(f"{i}_L")
    labeled_ears.add(f"{i}_R")

# Individual ears
individual_ears = [
    "61_R", "70_L", "71_L", "71_R", "75_R", "82_L", "82_R", "94_L", "97_L", "97_R", 
    "163_L", "169_L", "80_R", "83_L", "88_L", "152_L", "160_L", "166_L", "166_R", 
    "200_L", "231_L", "246_L", "270_L", "284_L", "289_L", "289_R", "298_L"
]
labeled_ears.update(individual_ears)

# Create mapping of all ears to their scores
all_ears_by_score = defaultdict(list)
labeled_ears_by_score = defaultdict(list)

for idx, row in df.iterrows():
    baby_id = row['baby_id']
    
    # Left ear
    l_ear = f"{baby_id}_L"
    l_score = row['L_score']
    all_ears_by_score[l_score].append(l_ear)
    if l_ear in labeled_ears:
        labeled_ears_by_score[l_score].append(l_ear)
    
    # Right ear
    r_ear = f"{baby_id}_R"
    r_score = row['R_score']
    all_ears_by_score[r_score].append(r_ear)
    if r_ear in labeled_ears:
        labeled_ears_by_score[r_score].append(r_ear)

# Target: 40 ears per category for scores 0-4
target_per_category = 40
target_scores = [0, 1, 2, 3, 4]

print("=" * 80)
print("LABELING PLAN TO REACH 40 EARS PER CATEGORY (Scores 0-4)")
print("=" * 80)

all_needed_ears = []
summary_stats = []

for score in target_scores:
    labeled_count = len(labeled_ears_by_score[score])
    total_available = len(all_ears_by_score[score])
    needed = target_per_category - labeled_count
    
    print(f"\n{'='*80}")
    print(f"SCORE {score}")
    print(f"{'='*80}")
    print(f"Already labeled: {labeled_count}")
    print(f"Target:          {target_per_category}")
    print(f"Still needed:    {max(0, needed)}")
    print(f"Total available: {total_available}")
    
    summary_stats.append({
        'score': score,
        'labeled': labeled_count,
        'needed': max(0, needed),
        'available': total_available
    })
    
    if needed > 0:
        # Get unlabeled ears in this category
        unlabeled = [ear for ear in all_ears_by_score[score] if ear not in labeled_ears]
        
        if len(unlabeled) < needed:
            print(f"\n⚠️  WARNING: Only {len(unlabeled)} unlabeled ears available, but need {needed}")
            print(f"   You can only label {len(unlabeled)} more ears in this category.")
            ears_to_label = unlabeled
        else:
            # Randomly select ears to label
            random.seed(42)  # For reproducibility
            ears_to_label = random.sample(unlabeled, needed)
        
        # Sort by baby_id for easier reference
        ears_to_label.sort(key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
        
        all_needed_ears.extend(ears_to_label)
        
        print(f"\n📋 Ears to label (first 30 shown):")
        print("-" * 80)
        
        # Print in a nice format, 5 per line
        for i in range(0, min(30, len(ears_to_label)), 5):
            batch = ears_to_label[i:i+5]
            print("   " + ", ".join(f"{ear:8s}" for ear in batch))
        
        if len(ears_to_label) > 30:
            print(f"   ... and {len(ears_to_label) - 30} more")
        
        # Also save to a text file for this score
        with open(f'/Users/mansapatel/Downloads/BabyEar4k/to_label_score_{score}.txt', 'w') as f:
            f.write(f"Score {score} - Ears to Label ({len(ears_to_label)} ears)\n")
            f.write("=" * 80 + "\n\n")
            for ear in ears_to_label:
                f.write(f"{ear}\n")
        print(f"\n✓ Full list saved to: to_label_score_{score}.txt")
    
    elif needed == 0:
        print(f"\n✓ Target already met! You have exactly {target_per_category} labeled ears.")
    
    else:  # needed < 0, meaning we have more than target
        excess = -needed
        print(f"\n✓ Target exceeded! You have {excess} more than needed.")
        print(f"   You could use any {target_per_category} of your {labeled_count} labeled ears.")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n{'Score':<10} {'Labeled':<12} {'Target':<10} {'Needed':<10} {'Available':<12} {'Status'}")
print("-" * 80)

for stat in summary_stats:
    status = "✓ Done" if stat['needed'] == 0 else (f"✓ Excess" if stat['needed'] < 0 else "⚠ Need more")
    if stat['needed'] > 0 and stat['available'] < stat['labeled'] + stat['needed']:
        status = "⚠ Insufficient"
    
    print(f"{stat['score']:<10} {stat['labeled']:<12} {target_per_category:<10} "
          f"{max(0, stat['needed']):<10} {stat['available']:<12} {status}")

total_needed = sum(max(0, stat['needed']) for stat in summary_stats)
print(f"\n{'TOTAL NEW LABELS NEEDED:':<44} {total_needed}")

# Save complete list
if all_needed_ears:
    with open('/Users/mansapatel/Downloads/BabyEar4k/complete_labeling_list.txt', 'w') as f:
        f.write("Complete List of Ears to Label\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total: {len(all_needed_ears)} ears\n\n")
        
        for score in target_scores:
            ears_in_score = [ear for ear in all_needed_ears 
                           if ear in all_ears_by_score[score]]
            if ears_in_score:
                f.write(f"\nScore {score} ({len(ears_in_score)} ears):\n")
                f.write("-" * 80 + "\n")
                for ear in ears_in_score:
                    f.write(f"{ear}\n")
    
    print(f"\n✓ Complete labeling list saved to: complete_labeling_list.txt")

# Create a condensed format for easy copy-paste
print("\n" + "=" * 80)
print("CONDENSED FORMAT (for easy copy-paste)")
print("=" * 80)

for score in target_scores:
    ears_in_score = [ear for ear in all_needed_ears 
                   if ear in all_ears_by_score[score]]
    if ears_in_score:
        print(f"\nScore {score} ({len(ears_in_score)} ears):")
        print(", ".join(ears_in_score))
