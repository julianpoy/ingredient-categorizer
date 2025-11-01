"""
Translate training data to multiple languages using local NLLB model
This on GPU since Google translate or others have limited/priced API
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = 'data_cleaned.txt'
OUTPUT_FILE = 'data_augmented.txt'
MODEL_NAME = 'facebook/nllb-200-distilled-1.3B'

# Translation batch size - adjust based on GPU memory
BATCH_SIZE = 512

# Maximum length for translations
MAX_LENGTH = 128

LANGUAGE_CODES = {
    'it-it': 'ita_Latn',  # Italian
    'de-de': 'deu_Latn',  # German
    'uk-ua': 'ukr_Cyrl',  # Ukrainian
    'he': 'heb_Hebr',     # Hebrew
    'es-es': 'spa_Latn',  # Spanish
    'fr-fr': 'fra_Latn',  # French
    'ru-ru': 'rus_Cyrl',  # Russian
    'hu-hu': 'hun_Latn',  # Hungarian
    'da-dk': 'dan_Latn',  # Danish
    'zh-cn': 'zho_Hans',  # Chinese (Simplified)
    'pt-pt': 'por_Latn',  # Portuguese
    'nl': 'nld_Latn',     # Dutch
    'pl': 'pol_Latn',     # Polish
    'ja': 'jpn_Jpan',     # Japanese
    'lt': 'lit_Latn',     # Lithuanian
    'eu': 'eus_Latn',     # Basque
    'el': 'ell_Grek',     # Greek
    'fi': 'fin_Latn',     # Finnish
    'sv': 'swe_Latn',     # Swedish
    'ro': 'ron_Latn',     # Romanian
}

SOURCE_LANG = 'eng_Latn'

# ============================================================
# FUNCTIONS
# ============================================================

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    
    data = {'category': [], 'text': []}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '____' not in line:
                continue
            
            parts = line.split('____', 1)
            if len(parts) == 2:
                category, text = parts
                data['category'].append(category.strip())
                data['text'].append(text.strip())
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df):,} samples")
    print(f"Categories: {df['category'].nunique()}")
    
    return df


def get_lang_token_id(tokenizer, lang_code):
    try:
        # Method 1: lang_code_to_id attribute
        if hasattr(tokenizer, 'lang_code_to_id'):
            return tokenizer.lang_code_to_id[lang_code]
    except:
        pass
    
    try:
        # Method 2: convert_tokens_to_ids
        return tokenizer.convert_tokens_to_ids(lang_code)
    except:
        pass
    
    try:
        # Method 3: Direct token lookup
        return tokenizer.get_vocab()[lang_code]
    except:
        pass
    
    print(f"Warning: Could not find token ID for {lang_code}, using default")
    return None


def translate_batch(texts, tokenizer, model, src_lang, tgt_lang, device):
    tokenizer.src_lang = src_lang
    
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)
    
    forced_bos_token_id = get_lang_token_id(tokenizer, tgt_lang)
    
    with torch.no_grad():
        gen_kwargs = {
            'max_length': MAX_LENGTH,
            'num_beams': 1,
            'do_sample': False,
        }
        
        if forced_bos_token_id is not None:
            gen_kwargs['forced_bos_token_id'] = forced_bos_token_id
        
        outputs = model.generate(**inputs, **gen_kwargs)
    
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return translations


def augment_data(df, tokenizer, model, device):
    print("\n" + "="*60)
    print("STARTING TRANSLATION")
    print("="*60)
    
    all_data = []
    
    # Keep original English data
    for _, row in df.iterrows():
        all_data.append({
            'category': row['category'],
            'text': row['text']
        })
    
    print(f"Original English samples: {len(all_data):,}")
    
    # Translate to each target language
    for lang_code, nllb_code in LANGUAGE_CODES.items():
        print(f"\n{'='*60}")
        print(f"Translating to {lang_code.upper()} ({nllb_code})")
        print(f"{'='*60}")
        
        categories = df['category'].tolist()
        texts = df['text'].tolist()
        
        # Process in batches
        translated_texts = []
        error_count = 0
        
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Translating to {lang_code}"):
            batch_texts = texts[i:i+BATCH_SIZE]
            
            try:
                batch_translations = translate_batch(
                    batch_texts,
                    tokenizer,
                    model,
                    SOURCE_LANG,
                    nllb_code,
                    device
                )
                translated_texts.extend(batch_translations)
            except Exception as e:
                error_count += 1
                if error_count <= 3:
                    print(f"\nError in batch {i}: {e}")
                translated_texts.extend(batch_texts)
        
        if error_count > 0:
            print(f"\n{error_count} batches had errors")
        else:
            print(f"All batches translated successfully")
        
        # Add translated data
        for category, text in zip(categories, translated_texts):
            all_data.append({
                'category': category,
                'text': text
            })
        
        print(f"Total samples so far: {len(all_data):,}")
    
    return pd.DataFrame(all_data)


def save_data(df, filepath):
    print(f"\n{'='*60}")
    print("SAVING AUGMENTED DATA")
    print(f"{'='*60}")
    print(f"\nSaving to {filepath}...")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['category']}____{row['text']}\n")
    
    print(f"Saved {len(df):,} samples")


def print_statistics(df):
    print("\n" + "="*60)
    print("FINAL DATASET STATISTICS")
    print("="*60)
    
    print(f"\nTotal samples: {len(df):,}")
    print(f"Unique texts: {df['text'].nunique():,}")
    print(f"Categories: {df['category'].nunique()}")
    
    print("\nCategory distribution:")
    category_counts = df['category'].value_counts().sort_index()
    for category, count in category_counts.items():
        print(f"  {category:<15} {count:>8,} samples")
    
    print("\n" + "="*60)
    print("SAMPLE TRANSLATIONS")
    print("="*60)
    
    sample_category = df['category'].iloc[0]
    sample_texts = df[df['category'] == sample_category]['text'].head(min(22, len(LANGUAGE_CODES) + 1)).tolist()
    
    print(f"\nCategory: {sample_category}")
    print(f"Original (English): {sample_texts[0]}")
    print("\nTranslations:")
    
    lang_names = list(LANGUAGE_CODES.keys())
    for i, lang in enumerate(lang_names, 1):
        if i < len(sample_texts):
            print(f"  {lang:<8} {sample_texts[i]}")


def main():
    print("="*60)
    print("MULTILINGUAL DATA AUGMENTATION")
    print("="*60)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Target languages: {len(LANGUAGE_CODES)}")
    print(f"Batch size: {BATCH_SIZE}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"\nLoading translation model: {MODEL_NAME}")
    print("This may take a few minutes on first run...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SOURCE_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    print(f"\nTokenizer type: {type(tokenizer).__name__}")
    
    print("\nTesting translation...")
    try:
        test_result = translate_batch(
            ["tomato", "chicken"], 
            tokenizer, 
            model, 
            SOURCE_LANG, 
            'spa_Latn', 
            device
        )
        print(f"Test successful: tomato → {test_result[0]}")
    except Exception as e:
        print(f"Test failed: {e}")
        print("Continuing anyway...")
    
    df = load_data(INPUT_FILE)
    
    total_translations = len(df) * len(LANGUAGE_CODES)
    batches_per_lang = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    total_batches = batches_per_lang * len(LANGUAGE_CODES)
    estimated_minutes = total_batches * 0.5  # rough estimate: 0.5s per batch
    
    print(f"\n{'='*60}")
    print("TRANSLATION PLAN")
    print(f"{'='*60}")
    print(f"Original samples: {len(df):,}")
    print(f"Target languages: {len(LANGUAGE_CODES)}")
    print(f"Translations needed: {total_translations:,}")
    print(f"Total batches: {total_batches:,}")
    print(f"Final dataset size: {len(df) * (len(LANGUAGE_CODES) + 1):,} samples")
    print(f"Estimated time: ~{estimated_minutes:.0f} minutes")
    
    augmented_df = augment_data(df, tokenizer, model, device)
    
    save_data(augmented_df, OUTPUT_FILE)
    
    print_statistics(augmented_df)
    
    print("\n" + "="*60)
    print("AUGMENTATION COMPLETE!")
    print("="*60)
    print(f"\nOriginal data: {len(df):,} samples")
    print(f"Augmented data: {len(augmented_df):,} samples")
    print(f"Multiplication factor: {len(augmented_df) / len(df):.1f}x")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTranslation interrupted")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
