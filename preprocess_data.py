"""
Data preprocessing script to merge categories and balance dataset
- Removes duplicates
- Filters categories
- Maps categories
- Balances to minimum category count
"""
import pandas as pd
from collections import Counter
import random

# ============================================================
# CONFIGURATION
# ============================================================

# Category mappings: [source_category, target_category]
CATEGORY_MAPPINGS = [
    ["health", "nonfood"],
    ["home", "nonfood"],
    # Add more mappings here as needed, e.g.:
    # ["vitamins", "nonfood"],
    # ["pharmacy", "nonfood"],
]

# Final categories we want to keep
TARGET_CATEGORIES = [
    'produce',
    'dairy', 
    'meat',
    'bakery',
    'grocery',
    'liquor',
    'seafood',
    'nonfood',
    'frozen',
    'canned',
    'beverages',
]

# Number of samples per category (None = use minimum category count)
SAMPLES_PER_CATEGORY = None  # Set to a number like 2164 to override

# Input/output files
INPUT_FILE = 'data.txt'
OUTPUT_FILE = 'data_cleaned.txt'

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================
# SCRIPT
# ============================================================

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    
    data = {'category': [], 'text': []}
    line_count = 0
    error_count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            line = line.strip()
            
            if not line or '____' not in line:
                continue
            
            try:
                parts = line.split('____', 1)
                if len(parts) != 2:
                    error_count += 1
                    continue
                    
                category, text = parts
                category = category.strip()
                text = text.strip()
                
                if text:
                    data['category'].append(category)
                    data['text'].append(text)
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"  Error parsing line {line_count}: {str(e)}")
    
    df = pd.DataFrame(data)
    print(f"  Loaded {len(df):,} valid samples from {line_count:,} lines")
    if error_count > 0:
        print(f"  Skipped {error_count:,} invalid lines")
    
    return df


def apply_category_mappings(df, mappings):
    if not mappings:
        print("\nNo category mappings to apply")
        return df
    
    print(f"\nApplying {len(mappings)} category mappings...")
    
    mapping_dict = {}
    for source, target in mappings:
        mapping_dict[source] = target
        source_count = (df['category'] == source).sum()
        print(f"  {source} -> {target}: {source_count:,} samples")
    
    df['category'] = df['category'].replace(mapping_dict)
    
    return df


def filter_categories(df, target_categories):
    print(f"\nFiltering to {len(target_categories)} target categories...")
    
    original_count = len(df)
    df = df[df['category'].isin(target_categories)].copy()
    filtered_count = original_count - len(df)
    
    print(f"  Kept {len(df):,} samples")
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count:,} samples from other categories")
    
    return df


def remove_duplicates(df):
    print("\nRemoving duplicates...")
    
    original_count = len(df)
    df = df.drop_duplicates(subset=['text', 'category']).copy()
    removed_count = original_count - len(df)
    
    print(f"  Removed {removed_count:,} duplicates ({removed_count/original_count*100:.2f}%)")
    print(f"  Remaining: {len(df):,} samples")
    
    return df


def balance_dataset(df, samples_per_category=None):
    print("\n" + "="*60)
    print("BALANCING DATASET")
    print("="*60)
    
    category_counts = df['category'].value_counts().sort_index()
    
    print("\nBefore balancing:")
    for category, count in category_counts.items():
        print(f"  {category:<15} {count:>8,} samples")
    
    min_count = category_counts.min()
    if samples_per_category is None:
        samples_per_category = min_count
        print(f"\nUsing minimum category count: {samples_per_category:,} samples per category")
    else:
        print(f"\nUsing specified: {samples_per_category:,} samples per category")
        if samples_per_category > min_count:
            print(f"  WARNING: Requested {samples_per_category:,} but smallest category has only {min_count:,}")
            print(f"  Using {min_count:,} instead")
            samples_per_category = min_count
    
    random.seed(RANDOM_SEED)
    balanced_dfs = []
    
    for category in sorted(df['category'].unique()):
        category_df = df[df['category'] == category]
        
        if len(category_df) >= samples_per_category:
            sampled = category_df.sample(n=samples_per_category, random_state=RANDOM_SEED)
        else:
            print(f"  WARNING: {category} has only {len(category_df):,} samples (requested {samples_per_category:,})")
            sampled = category_df
        
        balanced_dfs.append(sampled)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print("\nAfter balancing:")
    balanced_counts = balanced_df['category'].value_counts().sort_index()
    for category, count in balanced_counts.items():
        print(f"  {category:<15} {count:>8,} samples")
    
    print(f"\nTotal samples in balanced dataset: {len(balanced_df):,}")
    
    return balanced_df


def save_data(df, filepath):
    print(f"\nSaving data to {filepath}...")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['category']}____{row['text']}\n")
    
    print(f"  Saved {len(df):,} samples")


def print_statistics(df, title="DATASET STATISTICS"):
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    print(f"\nTotal samples: {len(df):,}")
    print(f"Unique items: {df['text'].nunique():,}")
    print(f"Categories: {df['category'].nunique()}")
    
    print("\nCategory distribution:")
    category_counts = df['category'].value_counts().sort_index()
    max_count = category_counts.max()
    
    for category, count in category_counts.items():
        percentage = (count / len(df)) * 100
        bar_length = int((count / max_count) * 30)
        bar = '█' * bar_length
        print(f"  {category:<15} {count:>8,} ({percentage:>5.2f}%) {bar}")
    
    if len(category_counts) > 1:
        min_count = category_counts.min()
        max_count = category_counts.max()
        ratio = max_count / min_count
        print(f"\nBalance ratio: {ratio:.2f}x")
        if ratio <= 1.1:
            print("  ✓ Dataset is well balanced")
        elif ratio <= 2:
            print("  Dataset is moderately balanced")
        else:
            print("  Dataset is imbalanced")
    
    df['text_length'] = df['text'].str.len()
    print(f"\nText length statistics:")
    print(f"  Average: {df['text_length'].mean():.1f} characters")
    print(f"  Median: {df['text_length'].median():.1f} characters")
    print(f"  Min: {df['text_length'].min()} characters")
    print(f"  Max: {df['text_length'].max()} characters")
    
    print("\nSample items per category:")
    for category in sorted(df['category'].unique()):
        category_items = df[df['category'] == category]['text']
        samples = category_items.sample(min(2, len(category_items)), random_state=RANDOM_SEED)
        print(f"\n{category.upper()}:")
        for item in samples:
            print(f"  • {item}")


def main():
    """Main preprocessing pipeline"""
    print("="*60)
    print("GROCERY DATA PREPROCESSING")
    print("="*60)
    print(f"\nInput: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Random seed: {RANDOM_SEED}")
    
    print("\nConfiguration:")
    print(f"  Category mappings: {len(CATEGORY_MAPPINGS)}")
    for source, target in CATEGORY_MAPPINGS:
        print(f"    {source} -> {target}")
    print(f"  Target categories: {len(TARGET_CATEGORIES)}")
    for cat in TARGET_CATEGORIES:
        print(f"    • {cat}")
    if SAMPLES_PER_CATEGORY:
        print(f"  Samples per category: {SAMPLES_PER_CATEGORY:,}")
    else:
        print(f"  Samples per category: Auto (use minimum)")
    
    df = load_data(INPUT_FILE)
    print_statistics(df, "ORIGINAL DATASET")
    
    df = apply_category_mappings(df, CATEGORY_MAPPINGS)
    
    df = filter_categories(df, TARGET_CATEGORIES)
    
    df = remove_duplicates(df)
    
    print_statistics(df, "AFTER FILTERING")
    
    df = balance_dataset(df, SAMPLES_PER_CATEGORY)
    
    print_statistics(df, "FINAL DATASET")
    
    save_data(df, OUTPUT_FILE)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"Total samples: {len(df):,}")
    print(f"Ready for training!")


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError:
        print(f"\nError: {INPUT_FILE} not found!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
