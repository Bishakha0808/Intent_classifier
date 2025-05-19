import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('user_intent_dataset.csv')  # Ensure file is in the same directory
    df = df.dropna(subset=['text', 'intent'])
    return df

df = load_data()

# 2. Intent keywords
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

# 3. Load Sentence-BERT model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# 4. Representative embeddings for fallback
rep_texts = df['text'].tolist()
rep_labels = df['intent'].tolist()
rep_embeddings = model.encode(rep_texts)

# 5. Preprocessing
def preprocess(text):
    return text.lower().strip()

# 6. Rule-based matching
def rule_based_match(text):
    text = preprocess(text)
    for intent, keywords in intent_keywords.items():
        for keyword in keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', text):
                return intent
    return None

# 7. BERT-based fallback
CONFIDENCE_THRESHOLD = 0.55

def bert_fallback(text):
    embedding = model.encode([text])
    similarity = cosine_similarity(embedding, rep_embeddings)
    best_idx = similarity.argmax()
    best_score = similarity[0, best_idx]

    if best_score >= CONFIDENCE_THRESHOLD:
        return rep_labels[best_idx] + f" (bert-fallback, score={best_score:.2f})"
    else:
        return "unknown (low confidence)"

# 8. Hybrid prediction
def hybrid_predict(text):
    rule_intent = rule_based_match(text)
    if rule_intent:
        return rule_intent + " (rule-based)"
    else:
        return bert_fallback(text)

# 9. Streamlit UI
st.title("Intent Detection App")
st.markdown("Enter a query below to classify user intent.")

user_input = st.text_input("Your input text:", "")
if st.button("Detect Intent"):
    if user_input.strip():
        prediction = hybrid_predict(user_input)
        st.success(f"*Predicted Intent:* {prediction}")
    else:
        st.warning("Please enter some text before clicking 'Detect Intent'.")