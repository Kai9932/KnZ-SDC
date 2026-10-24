import sys
from pathlib import Path

import joblib
import pymupdf
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = Path(__file__).resolve().parent.parent
MODEL = BASE / "models" / "distilbert_resume_classifier_v2"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
le = joblib.load(MODEL / "label_encoder.pkl")

print("classes in label encoder:", len(le.classes_))
print("classes in model:        ", model.config.num_labels)
print("random-guess confidence: ", round(100 / len(le.classes_), 1), "%")

with pymupdf.open(sys.argv[1]) as pdf:
    text = " ".join(" ".join(p.get_text().split()) for p in pdf)

def show(title, probs):
    print("\n" + title)
    top = torch.topk(probs, 5)
    for p, i in zip(top.values, top.indices):
        print(f"  {le.classes_[int(i)]:<30} {float(p) * 100:5.1f}%")

# A) first 256 tokens only (your original method)
enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
with torch.inference_mode():
    show("First 256 tokens only", torch.softmax(model(**enc).logits, -1)[0])

# B) every chunk, one by one
enc = tok(text, return_tensors="pt", truncation=True, max_length=256,
          stride=32, return_overflowing_tokens=True, padding=True)
enc.pop("overflow_to_sample_mapping", None)
with torch.inference_mode():
    probs = torch.softmax(model(**enc).logits, -1)
for n, p in enumerate(probs[:8]):
    show(f"Chunk {n + 1}", p)
show("Chunks averaged", probs[:8].mean(dim=0))