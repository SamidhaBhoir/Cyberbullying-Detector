from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

import json
import pickle
import re
import requests
import os
import numpy as np
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load LSTM model and artifacts (ensure files exist)
for _required_file in ['lstm_bully_model.keras', 'lstm_tokenizer.pkl', 'lstm_meta.pkl']:
    if not os.path.exists(_required_file):
        raise FileNotFoundError(f"Required file not found: {_required_file}. Run bully_detection.ipynb first.")

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model = tf.keras.models.load_model('lstm_bully_model.keras')

    with open('lstm_tokenizer.pkl', 'rb') as _f:
        tokenizer = pickle.load(_f)

    with open('lstm_meta.pkl', 'rb') as _f:
        meta = pickle.load(_f)

    MAX_SEQ_LEN = meta['MAX_SEQ_LEN']

except Exception as e:
    raise RuntimeError(f"Failed to load LSTM model: {e}")


def clean_text(text: str) -> str:
    """Replicate the same preprocessing used during model training."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_bully_prob(text: str) -> float:
    """Return the bully probability (0-1) for the given text."""
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    prob = model.predict(padded, verbose=0)[0][0]
    return float(prob)

# Instagram credentials (move to env vars for security)
load_dotenv()
long_access_token = os.getenv("LONG_ACCESS_TOKEN")
if not long_access_token:
    raise ValueError("LONG_ACCESS_TOKEN not found in .env file")

@app.get("/")
async def hello_world():
    return {"message":"Hello, World"}


@app.get("/privacy_policy", response_class=HTMLResponse)
async def privacy_policy():
    with open("./privacy_policy.html", "r", encoding="utf-8") as file:
        privacy_policy_html = file.read()
    return privacy_policy_html


@app.api_route("/webhook", methods=["GET", "POST"])
async def webhook(request: Request):

    if request.method == "POST":
        try:
            body = await request.json()
            logger.info(f"Received webhook: {json.dumps(body, indent=4)}")
            # Extract comment text (adjust based on webhook structure)
            comment_text = body['entry'][0]['changes'][0]['value']['text']
            comment_id = body['entry'][0]['changes'][0]['value']['id']  # For hiding

            # Keywords that are always considered bullying/inappropriate
            explicit_keywords = ['sex', 'xxx', 'porn', 'fuck', 'shit', 'ass', 'bitch', 'asshole', 'damn', 'dick']
            comment_lower = comment_text.lower()
            
            # Check for explicit keywords first
            has_explicit = any(keyword in comment_lower for keyword in explicit_keywords)
            
            if has_explicit:
                prediction = 1  # Flag as bully
                logger.info(f"Comment: '{comment_text}' | Contains explicit keywords | Classification: BULLY (explicit)")
            else:
                # Preprocess and classify using LSTM model
                bully_prob = predict_bully_prob(comment_text)

                # Lower threshold for more sensitivity (0.4 instead of 0.5)
                prediction = 1 if bully_prob > 0.4 else 0

                logger.info(f"Comment: '{comment_text}' | Bully prob: {bully_prob:.3f} | Classification: {'BULLY' if prediction == 1 else 'NOT BULLY'}")

            if prediction == 1:  # Bully
                # Hide the comment via Instagram API
                url = f"https://graph.facebook.com/v25.0/{comment_id}"
                data = {"hide": True, "access_token": long_access_token}
                response = requests.post(url, json=data)
                logger.info(f"Comment hidden: {response.json()}")

            return HTMLResponse("<p>Comment processed!</p>")

        except Exception as e:
            logger.error(f"Error processing comment: {e}")
            return HTMLResponse(f"<p>Error: {str(e)}</p>")
        
        return HTMLResponse("<p>Comment processed successfully!</p>")

    if request.method == "GET":
        params = request.query_params

        hub_mode = params.get("hub.mode")
        hub_challenge = params.get("hub.challenge")
        hub_verify_token = params.get("hub.verify_token")

        if hub_challenge:
            return PlainTextResponse(hub_challenge)

        return HTMLResponse("<p>This is GET Request, Hello Webhook!</p>")