import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from config import TFIDF_VECTORIZER_PATH, TFIDF_MODEL_PATH
from app.nlp.preprocessing import basic_clean
from app.utils.data_utils import ensure_dataset

def load_data():
    csv_path = ensure_dataset()
    df = pd.read_csv(csv_path)
    df["subject"] = df.get("subject", "")
    df["body"] = df.get("body", "")
    df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()
    df["text"] = df["text"].apply(basic_clean)
    return df[["text", "label"]]

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf = LogisticRegression(C=1.5, max_iter=200, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    print(classification_report(y_test, y_pred))
    TFIDF_VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    joblib.dump(clf, TFIDF_MODEL_PATH)
    print("Saved TF-IDF vectorizer and Logistic Regression model.")

if __name__ == "__main__":
    main()
