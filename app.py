from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np
import shap
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = FastAPI(title="Fraud Detection API")

# -----------------------------
# Load artifacts
# -----------------------------
model = joblib.load("Models/fraud_model_final.joblib")
explainer = joblib.load("Models/shap_explainer.joblib")

with open("Models/model_metadata.json", "r") as f:
    metadata = json.load(f)

threshold = metadata["threshold"]
feature_names = metadata["feature_cols"]

# -----------------------------
# LLM client (Gemini)
# -----------------------------
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

last_context = {}


# -----------------------------
# Case note generator (with fallback if Gemini fails)
# -----------------------------
def generate_case_note(prompt, top_features, probability, prediction):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        risk = "HIGH" if prediction else "LOW"
        reasons = "; ".join([
            f"{f['feature']} ({f['direction']})" for f in top_features[:3]
        ])
        return (
            f"[Fallback note — LLM unavailable] Risk level: {risk}. "
            f"Fraud probability: {probability:.4f}. "
            f"Top drivers: {reasons}. "
            f"Recommended action: {'escalate for manual review' if prediction else 'approve'}."
        )


# -----------------------------
# Request Schemas
# -----------------------------
class PredictionRequest(BaseModel):
    features: list[float]


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


# -----------------------------
# Home
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "model_type": metadata["model_type"]
    }


# -----------------------------
# Predict
# -----------------------------
@app.post("/predict")
def predict(request: PredictionRequest):

    if len(request.features) != len(feature_names):
        return {
            "error": f"Expected {len(feature_names)} features but received {len(request.features)}"
        }

    x = np.array(request.features).reshape(1, -1)
    probability = model.predict_proba(x)[0][1]
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "fraud_probability": float(probability),
        "threshold": threshold
    }


# -----------------------------
# Explain
# -----------------------------
@app.post("/explain")
def explain(request: PredictionRequest):

    if len(request.features) != len(feature_names):
        return {
            "error": f"Expected {len(feature_names)} features but received {len(request.features)}"
        }

    x = np.array(request.features).reshape(1, -1)

    probability = model.predict_proba(x)[0][1]
    prediction = int(probability >= threshold)

    shap_values = explainer.shap_values(x)[0]

    contributions = list(zip(feature_names, request.features, shap_values))
    contributions.sort(key=lambda item: abs(item[2]), reverse=True)

    top_features = [
        {
            "feature": name,
            "value": value,
            "shap_contribution": round(float(shap_val), 4),
            "direction": "increases fraud risk" if shap_val > 0 else "decreases fraud risk"
        }
        for name, value, shap_val in contributions[:5]
    ]

    features_text = "\n".join([
        f"- {f['feature']} = {f['value']} ({f['direction']}, impact score: {f['shap_contribution']})"
        for f in top_features
    ])

    prompt = (
        "You are a fraud analyst assistant. Write a short case note for a human reviewer "
        "based ONLY on the data given below. Never invent reasons that aren't in the data. "
        "Keep it to 3-4 sentences: state the risk level, the top drivers in plain English, "
        "and a recommended action (approve / escalate for review / decline).\n\n"
        f"Fraud probability: {probability:.4f}\n"
        f"Flagged: {'Yes' if prediction else 'No'}\n\n"
        f"Top contributing factors:\n{features_text}"
    )

    case_note = generate_case_note(prompt, top_features, probability, prediction)

    last_context["default"] = prompt

    return {
        "prediction": prediction,
        "fraud_probability": float(probability),
        "threshold": threshold,
        "top_features": top_features,
        "case_note": case_note
    }


# -----------------------------
# Chat
# -----------------------------
# -----------------------------
# Chat
# -----------------------------
@app.post("/chat")
def chat(request: ChatRequest):

    context = last_context.get(request.session_id)

    if not context:
        return {
            "answer": "No transaction has been explained yet. Call /explain first, then ask a follow-up question."
        }

    prompt = (
        "You are answering a fraud analyst's follow-up question about a flagged transaction. "
        "Only use the data below — do not guess or invent explanations beyond what's given. "
        "If the question can't be answered from this data, say so clearly.\n\n"
        f"Transaction data:\n{context}\n\n"
        f"Analyst question: {request.question}"
    )

    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        # real fallback: try to answer from the raw context text directly, no LLM needed
        question_lower = request.question.lower()

        if "threshold" in question_lower:
            answer = f"[Fallback — LLM unavailable] The fraud threshold is {threshold:.6f}."
        elif "probability" in question_lower or "score" in question_lower:
            answer = f"[Fallback — LLM unavailable] Here is the raw transaction data used for this case:\n\n{context}"
        else:
            answer = (
                "[Fallback — LLM unavailable] I can't generate a custom answer right now, "
                "but here is the full transaction context this case was based on:\n\n"
                f"{context}"
            )

    return {"answer": answer}