# 📱 SMS Spam Detection using Machine Learning

## 📌 Overview

This project implements an **SMS Spam Detection System** using Machine Learning techniques.
The goal is to automatically classify SMS messages as **Spam** or **Ham (Not Spam)** based on their textual content.

Such a system is crucial in preventing fraudulent, advertising, and malicious messages from reaching users.

---

## 🚀 Features

* Preprocessing of SMS text (cleaning, tokenization, stopword removal).
* Feature extraction using **TF-IDF / Bag of Words**.
* Model training using ML algorithms:

  * Naive Bayes
  * Support Vector Machine (SVM)
  * Random Forest
* Evaluation with **accuracy, precision, recall, F1-score**.
* Predict function to classify new SMS messages.

---

## 🗂️ Dataset

We used the **SMS Spam Collection Dataset** available from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

* Total messages: **5574**
* Labels:

  * `ham` → legitimate message
  * `spam` → unwanted message

---

## ⚙️ Tech Stack

* **Programming Language**: Python 🐍
* **Libraries**:

  * Pandas, NumPy → data handling
  * Scikit-learn → ML algorithms & evaluation
  * NLTK → text preprocessing
  * Matplotlib / Seaborn → visualization

---

## 📊 System Architecture

```
Dataset → Data Preprocessing → Feature Extraction → Model Training → Evaluation → Prediction
```

---

## 🔑 Methodology

1. **Data Cleaning & Preprocessing**

   * Removed punctuation, stopwords, and converted text to lowercase.
   * Tokenization and stemming applied.

2. **Feature Engineering**

   * Converted SMS text into numerical vectors using **TF-IDF Vectorizer**.

3. **Model Training**

   * Applied multiple algorithms (Naive Bayes, SVM, Random Forest).
   * Tuned hyperparameters for better performance.

4. **Evaluation**

   * Compared models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

---

## 📈 Results & Analysis

* **Naive Bayes** performed best for spam detection with high recall.
* Example evaluation:

  * Accuracy: **97%**
  * Precision: **96%**
  * Recall: **95%**
  * F1-Score: **95.5%**

---

## 🧪 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sms-spam-detection.git
cd sms-spam-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the training script

```bash
python train_model.py
```

### 4. Test with your own SMS

```bash
python predict.py "Congratulations! You won a free ticket. Call now!"
```

Output:

```
Prediction: Spam
```

---

## 📌 Future Scope

* Deploy the model using **Flask/Django API**.
* Integrate with a **mobile application** for real-time SMS filtering.
* Experiment with **deep learning models (LSTMs, BERT)** for better accuracy.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo and submit a pull request.

---

## 📜 License

This project is licensed under the **MIT License**.

---

✨ Developed with ❤️ using Machine Learning

