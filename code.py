import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load your dataset
df = pd.read_csv('/content/user_intent_dataset.csv')
df = df.dropna(subset=['text', 'intent'])

# 2. Define a keyword dictionary per intent
intent_keywords = {
    "confused about career": [
        "lost", "career", "confused", "don't know", "undecided", "which job", "what career"
    ],
    "emotionally stressed or demotivated": [
        "no motivation", "demotivated", "tired", "stressed", "burned out", "sad", "unproductive"
    ],
    "needs skill roadmap": [
        "learn", "skills", "roadmap", "how to become", "what to study", "study path", "need to learn"
    ],
    "asking for internship/job guidance": [
        "internship", "job", "apply", "get hired", "how to get", "job hunting", "resume"
    ]
}

# 3. Sentence-BERT model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 4. Preprocessing
def preprocess(text):
    return text.lower().strip()

# 5. Rule-based matching
def rule_based_match(text):
    text = preprocess(text)
    for intent, keywords in intent_keywords.items():
        for keyword in keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', text):
                return intent
    return None  # No match

# 6. BERT-based fallback
# Prepare a dictionary of canonical examples (or use avg of dataset per class)
# 6. BERT-based fallback with confidence threshold
CONFIDENCE_THRESHOLD = 0.55  # Recommended: 0.5–0.65; tune based on validation

def bert_fallback(text):
    embedding = model.encode([text])
    similarity = cosine_similarity(embedding, rep_embeddings)
    best_idx = similarity.argmax()
    best_score = similarity[0, best_idx]

    if best_score >= CONFIDENCE_THRESHOLD:
        return rep_labels[best_idx] + f" (bert-fallback, score={best_score:.2f})"
    else:
        return "unknown (low confidence)"

# 7. Hybrid prediction function with thresholded fallback
def hybrid_predict(text):
    rule_intent = rule_based_match(text)
    if rule_intent:
        return rule_intent + " (rule-based)"
    else:
        return bert_fallback(text)


# 7. Hybrid prediction function
def hybrid_predict(text):
    rule_intent = rule_based_match(text)
    if rule_intent:
        return rule_intent + " (rule-based)"
    else:
        return bert_fallback(text) + " (bert-fallback)"

# 8. Inference
test_inputs = [
    "I wanted to be a doctor",
    "Feeling very burnt out lately.",
    "How can I get an ML internship?",
    "What should I study to be a machine learning engineer?"
]

for sent in test_inputs:
    print(f"\nInput: {sent}")
    print(f"Predicted Intent: {hybrid_predict(sent)}")