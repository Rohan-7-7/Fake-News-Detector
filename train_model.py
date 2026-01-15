import pandas as pd
import numpy as np
import re
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# --- 1. Setup Environment ---
print("⚙️ Setting up environment...")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# --- 2. Check for Dataset ---
dataset_filename = 'WELFAKE_Dataset.csv'

if not os.path.exists(dataset_filename):
    print(f"❌ Error: '{dataset_filename}' not found!")
    print("   Please unzip your file and place 'WELFAKE_Dataset.csv' in this folder.")
    exit()

# --- 3. Define Cleaning Function ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML
    text = re.sub(r'[^a-zA-Z\s!?]', '', text) # Keep punctuation like ! and ?
    tokens = text.split()
    # Stemming reduces "running" -> "run" to match patterns better
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- 4. Load & Process Data ---
print(f"⏳ Loading {dataset_filename}...")
df = pd.read_csv(dataset_filename)

print(f"   Original dataset size: {len(df)} rows")

# Handle missing values
df = df.dropna(subset=['label'])
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')

# Combine title and text for better context
df['content'] = df['title'] + ' ' + df['text']

# --- OPTIONAL: Sampling for Speed ---
# If you want to train on the FULL dataset (slower but more accurate), 
# comment out the next 3 lines.
if len(df) > 20000:
    print(f"⚠️ Sampling 20,000 rows for faster training...")
    df = df.sample(n=20000, random_state=42)

print("🧹 Cleaning text (this takes the longest time)...")
df['clean_content'] = df['content'].apply(clean_text)

# --- 5. Vectorization ---
print("🔤 Converting text to numbers (TF-IDF)...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['clean_content'])
y = df['label']

# --- 6. Train Models ---
print("🏋️ Training Ensemble Model (Logistic Regression + Random Forest + SVM)...")

lr = LogisticRegression(solver='liblinear', random_state=42)
rf = RandomForestClassifier(n_estimators=50, random_state=42)
# Probability=True is required for the confidence score in the app
svc = SVC(kernel='linear', probability=True, random_state=42)

ensemble = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('svc', svc)],
    voting='soft'
)
ensemble.fit(X, y)

# --- 7. Save Files ---
print("💾 Saving 'ensemble_model.pkl' and 'tfidf_vectorizer.pkl'...")
pickle.dump(ensemble, open('ensemble_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))

print("✅ DONE! You can now run 'streamlit run app.py'")