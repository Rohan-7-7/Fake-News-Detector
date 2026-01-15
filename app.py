import pickle
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

try:
    model = pickle.load(open("ensemble_model.pkl", 'rb'))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", 'rb'))
except FileNotFoundError:
    st.error("🚨 Critical Error: Model files not found!")
    st.info("Please run 'train_model.py' first to generate 'ensemble_model.pkl' and 'tfidf_vectorizer.pkl'.")
    st.stop()


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'<.*?>', '', text)
   
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)

    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(102, 126, 234, 0.4);
    }
    .title-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .title-text {
        color: black;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .subtitle-text {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
.info-box {
    background-color: #f8f9fa;
    border-left: 5px solid #667eea;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1.5rem 0;
}

.info-box h3 {
    color: #667eea;   /* Theme purple */
    font-weight: 700;
}

.info-box p {
    color: #000000;   /* Keep description black for readability */
}

    .result-box-success {
        background-color: #d4edda;
        padding: 1.5rem; 
        border-radius: 10px; 
        border-left: 5px solid #28a745;
    }
    .result-box-danger {
        background-color: #fff3cd; 
        padding: 1.5rem; 
        border-radius: 10px; 
        border-left: 5px solid #ff6b6b;
    }
    </style>
""", unsafe_allow_html=True)

# --- 5. UI LAYOUT ---
st.markdown("""
    <div class="title-container">
        <h1 class="title-text">🔍 Fake News Detector</h1>
        <p class="subtitle-text">AI-Powered Truth Verification System</p>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
        <div class="info-box">
            <h3>📰 How It Works</h3>
            <p>Our advanced machine learning model analyzes text patterns to detect potential fake news. 
            Simply paste any news article below.</p>
        </div>
    """, unsafe_allow_html=True)
    
    text = st.text_area(
        "Enter or paste your text here:",
        height=200,
        placeholder="Paste the news article here...",
        help="Enter the text you want to analyze"
    )
    
    if text:
        st.caption(f"📝 Character count: {len(text)}")
    
    if st.button("🔎 Analyze Text"):
        if text.strip():
            with st.spinner("🤖 Analyzing text patterns..."):
                # Clean and predict
                cleaned_text = clean_text(text)
                text_converted = tfidf.transform([cleaned_text])
                
                # Get prediction (0 or 1) and probability
                prediction = model.predict(text_converted)[0]
                probability = model.predict_proba(text_converted)[0]
                

                confidence = max(probability) * 100
                
                st.markdown("---")
                

                
                if prediction == 0:
                    st.success(f"✅ **AUTHENTIC CONTENT DETECTED** ({confidence:.1f}% Confidence)")
                    st.markdown("""
                        <div class="result-box-success">
                            <h3 style="color: #155724; margin-top: 0;">✓ This appears to be genuine content</h3>
                            <p style="color: #155724; margin-bottom: 0;">
                            Our analysis indicates this text shows characteristics of authentic news (neutral tone, factual language).
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"⚠️ **POTENTIAL FAKE NEWS DETECTED** ({confidence:.1f}% Confidence)")
                    st.markdown("""
                        <div class="result-box-danger">
                            <h3 style="color: #856404; margin-top: 0;">⚡ Scam Alert!</h3>
                            <p style="color: #856404; margin-bottom: 0;">
                            This text shows patterns commonly associated with fake news (sensationalism, clickbait, conspiracy keywords).
                            Please verify this information before sharing.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text to analyze!")


st.markdown("---")
st.markdown("""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center; margin-top: 2rem;">
        <small style="color: #666;">
        <strong>Disclaimer:</strong> This tool provides automated analysis based on text patterns. 
        It is not 100% accurate. Always cross-reference with trusted news sources.
        </small>
    </div>
""", unsafe_allow_html=True)