import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "Fatimata/tweet_sentiment_model"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Créer un dictionnaire de correspondance
label_mapping = {0: "positif", 1: "négatif", 2: "neutre"}


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()

    # Convertir la sortie numérique en étiquette textuelle
    sentiment_label = label_mapping[predicted_class]

    return {
        "sentiment": sentiment_label,
        "text": text
    }


interface = gr.Interface(
    fn=predict_sentiment,
    inputs="text",
    outputs="text",
    title="Classification de sentiment",
    description="Entrez un texte pour prédire le sentiment."
)

interface.launch()
