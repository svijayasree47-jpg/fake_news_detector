import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

# Load data
data = pd.read_csv("news_data.csv")

X = data["text"]
y = data["label"]

# Create pipeline (vectorizer + model)
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression())
])

# Train
model.fit(X, y)

# Save
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")