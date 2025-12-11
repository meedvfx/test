import streamlit as st
import re
import numpy as np
import pypdf
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# --- 1. GESTION DE LA MÉMOIRE (SESSION STATE) ---
if 'text1_content' not in st.session_state:
    st.session_state.text1_content = ""
if 'text2_content' not in st.session_state:
    st.session_state.text2_content = ""

# --- 2. FONCTIONS UTILITAIRES ---

def extract_text_from_pdf(uploaded_file):
    """Extrait le texte d'un PDF."""
    try:
        pdf_reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        st.error(f"Erreur PDF : {e}")
        return ""

def update_text1_from_pdf():
    if st.session_state.uploader1:
        st.session_state.text1_content = extract_text_from_pdf(st.session_state.uploader1)

def update_text2_from_pdf():
    if st.session_state.uploader2:
        st.session_state.text2_content = extract_text_from_pdf(st.session_state.uploader2)

def preprocess_text(text):
    """Nettoyage : minuscules, suppression ponctuation/chiffres."""
    text_lower = text.lower()
    text_cleaned = re.sub(r'[^a-z\s]', '', text_lower)
    text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
    return text_cleaned

# --- 3. CHARGEMENT DES MODÈLES (CACHÉ) ---

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_word2vec_model():
    print("Téléchargement du modèle GloVe (Word2Vec)...")
    return api.load("glove-wiki-gigaword-50")

# --- 4. FONCTION SPÉCIFIQUE WORD2VEC ---
def get_word2vec_embedding(text, model):
    words = preprocess_text(text).split()
    vectors = []
    for word in words:
        if word in model:
            vectors.append(model[word])
    
    if not vectors:
        return np.zeros(model.vector_size)
    
    return np.mean(vectors, axis=0)

# --- 5. INTERFACE ---
st.set_page_config(page_title="Comparateur NLP", layout="wide")
st.title("🔎 Détecteur de Similarité (TF-IDF, Word2Vec, BERT)")

st.divider()

# Choix du modèle
st.header("1. Choisissez l'algorithme")
model_choice = st.radio(
    "Méthode :",
    ('TF-IDF (Statistique)', 'Word2Vec (Sémantique simple)', 'Sentence-BERT (Sémantique avancée)'),
    horizontal=True
)

ngram_tuple = (1, 1)
if 'TF-IDF' in model_choice:
    ngram_max = st.selectbox("Options N-grams :", (1, 2, 3, 4), format_func=lambda x: f"{x}-grams")
    ngram_tuple = (1, ngram_max)

st.divider()

# Zones d'entrée
st.header("2. Documents")
col1, col2 = st.columns(2)

with col1:
    st.file_uploader("PDF 1", type="pdf", key="uploader1", on_change=update_text1_from_pdf)
    text1 = st.text_area("Texte 1", height=300, key="text1_content")

with col2:
    st.file_uploader("PDF 2", type="pdf", key="uploader2", on_change=update_text2_from_pdf)
    text2 = st.text_area("Texte 2", height=300, key="text2_content")

# --- 6. CALCUL ---
if st.button("Lancer l'analyse", type="primary"):
    c1, c2 = text1.strip(), text2.strip()
    
    if not (c1 and c2):
        st.warning("Veuillez remplir les deux textes.")
    else:
        score = 0.0 # Initialisation
        
        # === CAS 1 : TF-IDF ===
        if 'TF-IDF' in model_choice:
            st.subheader("📊 Résultats TF-IDF")
            try:
                docs = [preprocess_text(c1), preprocess_text(c2)]
                vec = TfidfVectorizer(ngram_range=ngram_tuple)
                matrix = vec.fit_transform(docs)
                score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
                
                st.metric("Score de Similarité", f"{score*100:.2f} %")
                
            except ValueError:
                st.warning("Erreur : Textes vides après nettoyage.")
                score = 0.0

        # === CAS 2 : WORD2VEC ===
        elif 'Word2Vec' in model_choice:
            st.subheader("🧠 Résultats Word2Vec (GloVe)")
            with st.spinner("Chargement du modèle Word2Vec en cours..."):
                w2v_model = load_word2vec_model()
            
            v1 = get_word2vec_embedding(c1, w2v_model)
            v2 = get_word2vec_embedding(c2, w2v_model)
            
            # Calcul Cosinus
            score = cosine_similarity([v1], [v2])[0][0]
            
            st.metric("Score Sémantique (Moyenne des mots)", f"{score*100:.2f} %")

        # === CAS 3 : SENTENCE-BERT ===
        elif 'Sentence-BERT' in model_choice:
            st.subheader("🤖 Résultats Sentence-BERT")
            with st.spinner("Chargement du modèle BERT..."):
                sbert = load_sbert_model()
                
            emb = sbert.encode([c1, c2])
            score = util.pytorch_cos_sim(emb[0], emb[1]).item()
            
            st.metric("Score Sémantique (Contextuel)", f"{score*100:.2f} %")

        # --- CORRECTION DU BUG D'AFFICHAGE ---
        # On force le score entre 0.0 et 1.0 pour éviter le crash de st.progress
        safe_score = min(max(float(score), 0.0), 1.0)
        st.progress(safe_score)

        if score > 0.8: st.error("🚨 Similitude très forte.")
        elif score > 0.5: st.warning("⚠️ Similitude modérée.")
        else: st.success("✅ Textes différents.")
