# Grocery Categorizer

A multilingual grocery item categorization API that automatically classifies grocery items into categories using machine learning.

## Features

- Multilingual support (21+ languages including English, Spanish, French, German, Italian, Portuguese, Dutch, Danish, Swedish, Finnish, Polish, Russian, Ukrainian, Romanian, Hungarian, Greek, Hebrew, Lithuanian, Basque, Chinese, Japanese)
- 11 product categories: produce, dairy, meat, bakery, grocery, liquor, seafood, nonfood, frozen, canned, beverages
- Fast REST API with batch processing support
- Fine-tuned sentence transformer model for accurate categorization

## Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation and Training

### 1. Preprocess the Data

Clean, filter, and balance the raw data:

```bash
python preprocess_data.py
```

This script:
- Loads raw data from `data.txt`
- Applies category mappings (e.g., merging "health" and "home" into "nonfood")
- Filters to target categories
- Removes duplicates
- Balances the dataset to equal samples per category
- Saves cleaned data to `data_cleaned.txt`

### 2. Augment with Translations (Optional but Recommended)

Translate the English data into 20+ languages for multilingual support:

```bash
python augment_translations.py
```

This script:
- Loads `data_cleaned.txt`
- Uses the NLLB translation model to translate all items into 20+ languages
- Saves augmented data to `data_augmented.txt`
- Note: Requires GPU for reasonable performance, can take 30-60 minutes

### 3. Train the Model

Train the categorization model:

```bash
python train_categorizer.py
```

This script:
- Loads `data_augmented.txt`
- Fine-tunes a multilingual sentence transformer model
- Saves the trained model to `./grocery_model/`

Expected data format for all files:
```
category____item text
produce____Fresh tomatoes
dairy____Whole milk
```

## Running the API Server

Start the API server:

```bash
python server.py --host 0.0.0.0 --port 8000 --workers 1
```

## API Usage

### Categorize Items

```bash
curl -X POST "http://localhost:8000/categorize" \
  -H "Content-Type: application/json" \
  -d '{
    "items": ["2 lbs chicken breast", "Fresh tomatoes", "Whole milk"]
  }'
```

Response:
```json
{
  "results": [
    {
      "item": "2 lbs chicken breast",
      "category": "meat",
      "confidence": 0.9842,
      "probabilities": {
        "meat": 0.9842,
        "produce": 0.0089,
        "..."
      }
    }
  ],
  "processing_time_ms": 45.2
}
```

### Get Available Categories

```bash
curl "http://localhost:8000/categories"
```

### Health Check

```bash
curl "http://localhost:8000/health"
```

## Testing Inference

Run the inference script to test the model with sample items:

```bash
python inference.py
```

This will test categorization on sample items in multiple languages.

## Model Architecture

- Base model: `paraphrase-multilingual-MiniLM-L12-v2` (118M parameters, supports 50+ languages)
- Custom classification head with 256 hidden units
- Fine-tuned on grocery item data
- Translation model (for augmentation): `facebook/nllb-200-distilled-1.3B`

