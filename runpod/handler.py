
import os
import pickle

import runpod
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# The Network Volume is mounted here by RunPod Serverless.
MODEL_DIR = "/runpod-volume/models/distilbert_resume_classifier_v2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading resume classifier from {MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

model.to(device)
model.eval()

encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
label_encoder = None

if os.path.exists(encoder_path):
    with open(encoder_path, "rb") as file:
        label_encoder = pickle.load(file)


def handler(job: dict) -> dict:
    data = job.get("input", {})
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return {"error": "Please provide non-empty resume text."}

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        predicted_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_id].item())

    if label_encoder is not None:
        predicted_role = str(
            label_encoder.inverse_transform([predicted_id])[0]
        )
    else:
        predicted_role = str(predicted_id)

    return {
        "predicted_role": predicted_role,
        "class_id": predicted_id,
        "confidence": confidence,
    }


runpod.serverless.start({"handler": handler})
