
import csv
from collections import defaultdict


# Read the CSV file
with open('/Users/mansapatel/Downloads/BabyEar4k/diagnosis_result.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Extract the first number from L_merge and R_merge columns
def get_first_value(cell_value):
    """Extract the first number from format 'X+Y+Z'"""
    return int(cell_value.split('+')[0])

# Organize ears by category
ears_by_category = defaultdict(list)

# Priority ranges
priority_ranges = [(1, 60), (100, 150)]

# First, collect ears from priority ranges
for range_start, range_end in priority_ranges:
    for row in data:
        baby_id = int(row['baby_id'])
        if range_start <= baby_id <= range_end:
            l_value = get_first_value(row['L_merge'])
            r_value = get_first_value(row['R_merge'])
            
            ears_by_category[l_value].append(f"{baby_id}_L")
            ears_by_category[r_value].append(f"{baby_id}_R")

# If we don't have enough, add from all other ranges
for category in range(5):
    if len(ears_by_category[category]) < 20:
        for row in data:
            baby_id = int(row['baby_id'])
            # Skip if already in priority ranges
            if any(start <= baby_id <= end for start, end in priority_ranges):
                continue
            
            l_value = get_first_value(row['L_merge'])
            r_value = get_first_value(row['R_merge'])
            
            if l_value == category and f"{baby_id}_L" not in ears_by_category[category]:
                ears_by_category[category].append(f"{baby_id}_L")
            if r_value == category and f"{baby_id}_R" not in ears_by_category[category]:
                ears_by_category[category].append(f"{baby_id}_R")
            
            if len(ears_by_category[category]) >= 20:
                break

# Display results
print("=" * 70)
print("20 Ear Samples from Each Category")
print("=" * 70)

for category in range(5):
    ears = ears_by_category[category][:20]
    print(f"\nCategory {category}: {len(ears)} ears found")
    print("-" * 70)
    
    # Print in rows of 5 for readability
    for i in range(0, len(ears), 5):
        row_ears = ears[i:i+5]
        print("  " + ", ".join(f"{ear:>8}" for ear in row_ears))
    
    if len(ears_by_category[category]) < 20:
        print(f"  ⚠️  Only {len(ears)} ears available (need 20)")
    
print("\n" + "=" * 70)

# Summary
print("\nSummary:")
for category in range(5):
    total = len(ears_by_category[category])
    print(f"Category {category}: {total:3d} total ears available")
print("=" * 70)