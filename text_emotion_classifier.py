"""
Clasificador de emociones en texto/voz
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import os

class TextEmotionClassifier:
    def __init__(self, model_path="./models/fine_tuned_beto"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"��️  TextClassifier usando: {self.device}")
        
        # Cargar modelo
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        with open(f"{model_path}/label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print(f"✅ Clasificador de texto cargado")
        print(f"   Emociones: {list(self.label_encoder.classes_)[:5]}...")
    
    def classify(self, text: str) -> dict:
        """
        Clasifica texto y retorna emociones
        """
        if not text or not text.strip():
            return {
                "emotions": {},
                "primary_emotion": "neutral",
                "confidence": 0.0
            }
        
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Crear diccionario de emociones
        emotions = {}
        for emotion, prob in zip(self.label_encoder.classes_, probabilities):
            if emotion and str(emotion) != 'nan':  # Filtrar emociones inválidas
                emotions[str(emotion)] = float(prob)
        
        # Encontrar emoción primaria
        if emotions:
            primary_emotion = max(emotions, key=emotions.get)
            confidence = emotions[primary_emotion]
        else:
            primary_emotion = "neutral"
            confidence = 0.0
        
        return {
            "emotions": emotions,
            "primary_emotion": primary_emotion,
            "confidence": confidence
        }
