from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import pickle, json

texts = [
    "Free money, claim now!",
    "Hi, how are you?",
    "Win a lottery today",
    "Let's meet for lunch"
]
labels = ["spam", "ham", "spam", "ham"]

# Train-test split
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    texts, labels, test_size=0.5, random_state=42, stratify=labels
)

# Fit vectorizer only on training data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train_txt)  # ONLY here
X_test = vectorizer.transform(X_test_txt)        # no fit here

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_train_proba = model.predict_proba(X_train)
y_test_proba = model.predict_proba(X_test)

# Metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_loss = log_loss(y_train, y_train_proba, labels=model.classes_)
test_loss = log_loss(y_test, y_test_proba, labels=model.classes_)

# Save model and vectorizer
with open("sms_spam_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

metrics = {
    "train_accuracy": train_accuracy,
    "test_accuracy": test_accuracy,
    "train_loss": train_loss,
    "test_loss": test_loss
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("✅ Model, vectorizer, and metrics saved successfully.")
print("📊 Metrics:", metrics)
