import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# --- 1. SETUP ---
print("⚙️ Setting up evaluation environment...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load Models
print("📂 Loading saved models...")
try:
    model = pickle.load(open("ensemble_model.pkl", 'rb'))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", 'rb'))
except FileNotFoundError:
    print("❌ Error: Model files not found. Run train_model.py first.")
    exit()

# Preprocessing Function (MUST MATCH TRAINING)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- 2. LOAD & PREPARE DATA ---
filename = 'WELFAKE_Dataset.csv'
print(f"⏳ Loading data from {filename}...")

# Load only a sample for testing (e.g., 5000 random rows) to keep it fast
# We use a different random_state than training to simulate "new" data
df = pd.read_csv(filename)
df = df.dropna(subset=['label'])
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')
df['content'] = df['title'] + ' ' + df['text']

# Sample 5,000 rows for testing
test_df = df.sample(n=5000, random_state=999) 

print("🧹 Cleaning test data...")
test_df['clean_content'] = test_df['content'].apply(clean_text)

# Prepare X and y
X_test = tfidf.transform(test_df['clean_content'])
y_test = test_df['label']

# --- 3. PREDICTIONS ---
print("🤖 Running predictions...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # Probability for Class 1

# --- 4. CALCULATE METRICS ---
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 5. PLOTTING ---
plt.figure(figsize=(15, 6))

# Plot 1: Confusion Matrix
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Pred: Real', 'Pred: Fake'],
            yticklabels=['Actual: Real', 'Actual: Fake'])
plt.title('Confusion Matrix\n(Where did the model make mistakes?)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# Plot 2: ROC Curve
plt.subplot(1, 2, 2)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve\n(Higher curve = Better performance)')
plt.legend(loc="lower right")

plt.tight_layout()
print("📊 Displaying graphs...")
plt.show()