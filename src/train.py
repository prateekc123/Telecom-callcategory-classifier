import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from data_preprocessing import preprocess_data, preprocess

def train_model(file_path):
    df=preprocess_data(file_path)
    df=preprocess(df)

    X=df['clean_text']
    y=df['label']

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    vectorizer = TfidfVectorizer(max_features=7000,ngram_range=(1,3),stop_words='english')
    X_train_vec=vectorizer.fit_transform(X_train)
    X_test_vec=vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000, class_weight='balanced')

    model.fit(X_train_vec,y_train)
    y_pred=model.predict(X_test_vec)


    print(accuracy_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))

    joblib.dump(model,'models/model.pkl')
    joblib.dump(vectorizer,'models/vectorizer.pkl')

if __name__ == "__main__":
    train_model("data/raw/textlabel3.csv")

