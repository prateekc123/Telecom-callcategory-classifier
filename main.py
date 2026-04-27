from src.predict import predict

while True:
    text=input("Enter your query: ")
    label, confidence=predict(text)
    print(f"Prediction: {label}, Confidence: {confidence:.2f}")