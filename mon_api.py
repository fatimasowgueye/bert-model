from fastapi import FastAPI
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()

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


@app.post("/predict_sentiment/")
async def classify_sentiment(text: str):
    result = predict_sentiment(text)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
