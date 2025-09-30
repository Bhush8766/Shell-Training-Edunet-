# backend/train_model.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Example training data
texts = [
    "Free money, claim now!",
    "Hi, how are you?",
    "Win a lottery today",
    "Let's meet for lunch"
]
labels = ["spam", "ham", "spam", "ham"]

# Train vectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X_train, labels)

# Save model and vectorizer
with open("sms_spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully.")
