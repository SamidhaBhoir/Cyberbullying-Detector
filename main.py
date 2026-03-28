from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

import json
import joblib
import requests
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model and vectorizer (ensure files exist)
if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
    raise FileNotFoundError("Model files (model.pkl, vectorizer.pkl) not found. Run api.ipynb first.")

try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

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
                # Preprocess and classify using model
                comment_vectorized = vectorizer.transform([comment_text])
                probabilities = model.predict_proba(comment_vectorized)[0]
                bully_prob = probabilities[1]  # Probability of being bully
                
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