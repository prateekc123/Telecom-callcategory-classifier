import joblib
import re

# -----------------------------
# Load model and vectorizer
# -----------------------------
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')


# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


# -----------------------------
# Rule-Based Override (Improved)
# -----------------------------
def rule_based_override(text):
    t = text.lower()

    # Retention
    if any(k in t for k in [
        "cancel", "disconnect", "switch", "leave",
        "terminate", "port", "moving to", "going to"
    ]):
        return "Retention"

    # Sales
    if any(k in t for k in [
        "price", "cost", "how much", "plan",
        "offer", "new connection", "sign up"
    ]):
        return "Sales"

    # Billing (light)
    if any(k in t for k in [
        "bill", "charge", "payment", "invoice"
    ]):
        return "Billing"

    # Repair (light)
    if any(k in t for k in [
        "not working", "issue", "problem", "network", "internet"
    ]):
        return "Repair"

    return None


# -----------------------------
# Prediction Function
# -----------------------------
def predict(text):
    # Step 1: Rule-based override (high confidence intents)
    rule = rule_based_override(text)
    if rule:
        return rule, 0.95

    # Step 2: ML prediction
    text_clean = clean_text(text)
    vec = vectorizer.transform([text_clean])

    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()

    # Step 3: Smarter threshold handling
    if prob < 0.4:
        return "Uncertain", prob

    # Medium confidence → still return prediction
    if 0.4 <= prob < 0.6:
        return f"{pred} (Low Confidence)", prob

    return pred, prob