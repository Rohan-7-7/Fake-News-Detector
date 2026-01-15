import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score

# --- 1. CONFIGURATION ---
print("⚙️ Setting up environment...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# --- 2. LOAD SAVED SYSTEM ---
print("📂 Loading your trained models...")
try:
    ensemble_model = pickle.load(open("ensemble_model.pkl", 'rb'))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", 'rb'))
except FileNotFoundError:
    print("❌ Error: Model files not found. Please run train_model.py first.")
    exit()

# Preprocessing function (Standardized)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- 3. PREPARE TEST DATA ---
filename = 'WELFAKE_Dataset.csv'
print(f"⏳ Loading test data from {filename}...")

# Load and sample data
df = pd.read_csv(filename)
df = df.dropna(subset=['label'])
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')
df['content'] = df['title'] + ' ' + df['text']

# Use a new random sample for testing
test_df = df.sample(n=5000, random_state=42) 

print("🧹 Cleaning and Vectorizing...")
test_df['clean_content'] = test_df['content'].apply(clean_text)
X_test = tfidf.transform(test_df['clean_content'])
y_test = test_df['label']

# --- 4. EVALUATE INDIVIDUAL MODELS VS ENSEMBLE ---
print("🧪 Testing individual algorithms...")

results = {}

# 1. Test the internal sub-models (LR, RF, SVM)
# The VotingClassifier stores its fitted sub-models in 'estimators_'
model_names = ['Logistic Regression', 'Random Forest', 'SVM (Linear)']
colors = ['#3498db', '#9b59b6', '#e67e22', '#2ecc71'] # Blue, Purple, Orange, Green

for name, model in zip(model_names, ensemble_model.estimators_):
    print(f"   Running {name}...")
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc * 100

# 2. Test the Main Ensemble
print("   Running Ensemble (Proposed System)...")
pred_ensemble = ensemble_model.predict(X_test)
acc_ensemble = accuracy_score(y_test, pred_ensemble)
results['Ensemble (Proposed)'] = acc_ensemble * 100

# --- 5. GENERATE THE GRAPH ---
print("📊 Plotting graph...")

# Convert to DataFrame for easier plotting
df_results = pd.DataFrame(list(results.items()), columns=['Algorithm', 'Accuracy'])

# Setup the Plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Create Bar Chart
ax = sns.barplot(x='Algorithm', y='Accuracy', data=df_results, palette=colors)

# Customization to match Research Paper style
plt.ylim(85, 100) # Zoom in on the top percentages to show differences
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
plt.title('Accuracy Comparison: Individual Models vs. Proposed Ensemble', fontsize=14, fontweight='bold', pad=20)

# Add the numbers on top of the bars
for i, v in enumerate(df_results['Accuracy']):
    ax.text(i, v + 0.2, f"{v:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.show()
print("✅ Done!")