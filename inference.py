import torch
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from train_categorizer import GroceryClassifier


class GroceryCategorizer:
    def __init__(self, model_dir='./grocery_model'):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with open(self.model_dir / 'label_mapping.json', 'r') as f:
            mapping = json.load(f)
            self.label2id = mapping['label2id']
            self.id2label = {int(k): v for k, v in mapping['id2label'].items()}
        
        print(f"Loading model from {model_dir}...")
        self.base_model = SentenceTransformer(str(self.model_dir / 'sentence_transformer'))
        
        embedding_dim = self.base_model.get_sentence_embedding_dimension()
        self.classifier = GroceryClassifier(embedding_dim, len(self.id2label))
        self.classifier.load_state_dict(
            torch.load(self.model_dir / 'classifier.pt', map_location=self.device)
        )
        
        self.base_model.to(self.device)
        self.classifier.to(self.device)
        self.base_model.eval()
        self.classifier.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def predict(self, text):
        with torch.no_grad():
            embedding = self.base_model.encode(
                [text],
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            output = self.classifier(embedding)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_id = torch.argmax(probabilities).item()
            predicted_category = self.id2label[predicted_id]
            
            probs_dict = {
                self.id2label[i]: probabilities[i].item()
                for i in range(len(self.id2label))
            }
            return {
                'category': predicted_category,
                'confidence': probabilities[predicted_id].item(),
                'probabilities': probs_dict
            }
    
    def predict_batch(self, texts):
        with torch.no_grad():
            embeddings = self.base_model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=32
            )
            
            outputs = self.classifier(embeddings)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_ids = torch.argmax(probabilities, dim=1)
            
            results = []
            for i, pred_id in enumerate(predicted_ids):
                pred_id = pred_id.item()
                category = self.id2label[pred_id]
                
                probs_dict = {
                    self.id2label[j]: probabilities[i][j].item()
                    for j in range(len(self.id2label))
                }
                results.append({
                    'category': category,
                    'confidence': probabilities[i][pred_id].item(),
                    'probabilities': probs_dict
                })
            
            return results


def main():
    categorizer = GroceryCategorizer('./grocery_model')
    
    test_items = [
        "1 (28 oz) can San Marzano whole peeled tomatoes",
        "Toilet paper",
        "Kind bars (julian)",
        "Face wash (TJ's)",
        "2 lbs chicken breast",
        "Fresh basil",
        "Whole milk",
        "Sourdough bread",
        "Red wine",
        "Atlantic salmon",
        "Sliced turkey (deli counter)",
        # Multilingual examples
        "2 kg tomates",  # French
        "500g Hähnchenbrust",  # German
        "Leche entera",  # Spanish
        "新鮮なバジル",  # Japanese
    ]
    
    print("\n" + "="*60)
    print("Testing Grocery Categorizer")
    print("="*60)
    
    print("\nDetailed predictions:")
    for item in test_items[:6]:
        result = categorizer.predict(item)
        print(f"\nItem: {item}")
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Top 3 probabilities:")
        sorted_probs = sorted(
            result['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        for cat, prob in sorted_probs:
            print(f"  {cat}: {prob:.4f}")
    
    print("\n" + "="*60)
    print("Batch predictions:")
    print("="*60)
    categories = categorizer.predict_batch(test_items)
    for item, category in zip(test_items, categories):
        print(f"{item:50s} -> {category}")
    
    print("\n" + "="*60)
    print("Testing multilingual capability:")
    print("="*60)
    multilingual_items = [
        ("English", "Fresh tomatoes"),
        ("Spanish", "Tomates frescos"),
        ("French", "Tomates fraîches"),
        ("German", "Frische Tomaten"),
        ("Italian", "Pomodori freschi"),
    ]
    
    for language, item in multilingual_items:
        category = categorizer.predict(item)
        print(f"{language:10s} | {item:25s} -> {category}")


if __name__ == '__main__':
    main()
