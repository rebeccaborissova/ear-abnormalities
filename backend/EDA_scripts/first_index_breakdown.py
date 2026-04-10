import pandas as pd
from collections import Counter

# Read the CSV file
df = pd.read_csv('/Users/mansapatel/Downloads/BabyEar4k/diagnosis_result.csv')

# Function to extract the first number from the score string
def get_first_index(score_str):
    """Extract the first number from a string like '7+0+1' -> 7"""
    return int(str(score_str).split('+')[0])

# Extract first index scores for left and right ears
left_scores = df['L01'].apply(get_first_index)
right_scores = df['R01'].apply(get_first_index)

# Count occurrences for each score (0-7)
left_counts = Counter(left_scores)
right_counts = Counter(right_scores)

# Print breakdown for Left Ear (L01)
print("=" * 50)
print("LEFT EAR (L01) - First Index Breakdown")
print("=" * 50)
total_left = len(left_scores)
for score in range(8):
    count = left_counts.get(score, 0)
    percentage = (count / total_left) * 100
    print(f"Score {score}: {count:4d} ears ({percentage:5.2f}%)")
print(f"\nTotal Left Ears: {total_left}")

print("\n")

# Print breakdown for Right Ear (R01)
print("=" * 50)
print("RIGHT EAR (R01) - First Index Breakdown")
print("=" * 50)
total_right = len(right_scores)
for score in range(8):
    count = right_counts.get(score, 0)
    percentage = (count / total_right) * 100
    print(f"Score {score}: {count:4d} ears ({percentage:5.2f}%)")
print(f"\nTotal Right Ears: {total_right}")

print("\n")

# Combined breakdown
print("=" * 50)
print("COMBINED (Both Ears) - First Index Breakdown")
print("=" * 50)
combined_scores = list(left_scores) + list(right_scores)
combined_counts = Counter(combined_scores)
total_combined = len(combined_scores)
for score in range(8):
    count = combined_counts.get(score, 0)
    percentage = (count / total_combined) * 100
    print(f"Score {score}: {count:4d} ears ({percentage:5.2f}%)")
print(f"\nTotal Ears (Both): {total_combined}")

# Check unique scores to see if there are any outside 0-7 range
unique_scores = set(combined_scores)
if any(score < 0 or score > 7 for score in unique_scores):
    print("\n⚠️  Note: Found scores outside 0-7 range:")
    print(f"All unique scores: {sorted(unique_scores)}")
