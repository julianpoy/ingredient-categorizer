import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from torch import nn
import json

CATEGORIES = [
    'produce', 'dairy', 'meat', 'bakery', 'grocery', 
    'liquor', 'seafood', 'nonfood', 'frozen', 'canned', 'beverages'
]
# Balance settings: 
# - Set to None to use all data
# - Set to a number to balance during training (e.g., 5000)
SAMPLES_PER_CATEGORY = None  
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2' # 118M params, supports 50+ languages
BATCH_SIZE = 700
EPOCHS = 5
LEARNING_RATE = 2e-5
# Format should be category____text
DATA_FILE = 'data_augmented.txt'

class GroceryDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def load_and_balance_data(data_path):
    print(f"Loading data from {data_path}...")
    
    data = {'category': [], 'text': []}
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '____' not in line:
                continue
            
            category, text = line.split('____', 1)
            if category in CATEGORIES and len(text) > 0:
                data['category'].append(category)
                data['text'].append(text)
    
    df = pd.DataFrame(data)
    print(f"\nOriginal dataset size: {len(df)}")
    print("\nCategory distribution:")
    print(df['category'].value_counts())
    
    if SAMPLES_PER_CATEGORY is None:
        print(f"\nSAMPLES_PER_CATEGORY is None - using all data without balancing")
        balanced_df = df
    else:
        print(f"\nBalancing to {SAMPLES_PER_CATEGORY} samples per category...")
        balanced_dfs = []
        
        for category in CATEGORIES:
            category_df = df[df['category'] == category]
            if len(category_df) >= SAMPLES_PER_CATEGORY:
                sampled = category_df.sample(n=SAMPLES_PER_CATEGORY, random_state=42)
            else:
                print(f"Warning: {category} has only {len(category_df)} samples")
                sampled = category_df
            balanced_dfs.append(sampled)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    print(f"\nBalanced dataset size: {len(balanced_df)}")
    print("\nBalanced distribution:")
    print(balanced_df['category'].value_counts())
    
    return balanced_df


class GroceryClassifier(nn.Module):
    """Classification head on top of sentence transformer"""
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, embeddings):
        return self.classifier(embeddings)


def train_model(df, output_dir='./grocery_model'):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    label2id = {label: idx for idx, label in enumerate(CATEGORIES)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    with open(output_dir / 'label_mapping.json', 'w') as f:
        json.dump({'label2id': label2id, 'id2label': id2label}, f, indent=2)
    
    # Prepare data
    df['label_id'] = df['category'].map(label2id)
    
    train_df, test_df = train_test_split(
        df, test_size=0.1, stratify=df['label_id'], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df['label_id'], random_state=42
    )
    
    print(f"\nTrain size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    print(f"\nLoading base model: {MODEL_NAME}")
    base_model = SentenceTransformer(MODEL_NAME)
    embedding_dim = base_model.get_sentence_embedding_dimension()
    
    classifier = GroceryClassifier(embedding_dim, len(CATEGORIES))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.to(device)
    base_model.to(device)
    
    print(f"Using device: {device}")
    
    train_dataset = GroceryDataset(train_df['text'].tolist(), train_df['label_id'].tolist())
    val_dataset = GroceryDataset(val_df['text'].tolist(), val_df['label_id'].tolist())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    optimizer = torch.optim.AdamW(
        list(base_model.parameters()) + list(classifier.parameters()),
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        base_model.train()
        classifier.train()
        train_loss = 0
        
        for texts, labels in train_loader:
            optimizer.zero_grad()
            
            features = base_model.tokenize(texts)
            features = {k: v.to(device) for k, v in features.items()}
            embeddings = base_model.forward(features)['sentence_embedding']
            
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.long, device=device)
            else:
                labels = labels.clone().detach().to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long, device=device)
            
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        base_model.eval()
        classifier.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for texts, labels in val_loader:
                embeddings = base_model.encode(
                    texts, 
                    convert_to_tensor=True, 
                    show_progress_bar=False
                )
                
                if isinstance(labels, list):
                    labels = torch.tensor(labels, dtype=torch.long, device=device)
                else:
                    labels = labels.to(device)
                
                outputs = classifier(embeddings)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += len(labels)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            base_model.save(str(output_dir / 'sentence_transformer'))
            torch.save(classifier.state_dict(), output_dir / 'classifier.pt')
            print(f"  Saved best model (val_acc: {val_acc:.4f})")
    
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)
    
    base_model = SentenceTransformer(str(output_dir / 'sentence_transformer'))
    classifier.load_state_dict(torch.load(output_dir / 'classifier.pt'))
    base_model.to(device)
    classifier.to(device)
    base_model.eval()
    classifier.eval()
    
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label_id'].tolist()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, len(test_texts), BATCH_SIZE):
            batch_texts = test_texts[i:i+BATCH_SIZE]
            batch_labels = test_labels[i:i+BATCH_SIZE]
            
            embeddings = base_model.encode(
                batch_texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            outputs = classifier(embeddings)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(batch_labels)
    
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_predictions,
        target_names=CATEGORIES,
        digits=4
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_predictions)
    print("Categories:", CATEGORIES)
    print(cm)
    
    np.save(output_dir / 'confusion_matrix.npy', cm)
    
    print(f"\nModel saved to {output_dir}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return base_model, classifier, label2id, id2label


if __name__ == '__main__':
    print(f"Using data file: {DATA_FILE}")
    df = load_and_balance_data(DATA_FILE)
    
    train_model(df)
    
    print("\nTraining complete!")
