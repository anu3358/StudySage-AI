# ============================================================
# 📚 StudySage AI — Smart Study Companion
# Author: ANURAG SAINI THE BAKU
# Description: Analyze past question papers using NLP (TF-IDF)
#              to predict important exam topics intelligently.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import nltk
import re
from PyPDF2 import PdfReader

# ------------------ Setup ------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('english'))

st.set_page_config(page_title='StudySage AI', layout='wide')


# ------------------ Helper Functions ------------------
def read_pdf(file) -> str:
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception:
        return ""


def clean_text(text: str) -> str:
    """Cleans and tokenizes raw text."""
    if text is None:
        return ""
    text = text.replace("\n", " ").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2 and not t.isdigit()]
    return " ".join(tokens)


def extract_text_from_upload(uploaded_file):
    """Handles both PDF and TXT uploads."""
    fname = uploaded_file.name.lower()
    if fname.endswith(".pdf"):
        return read_pdf(uploaded_file)
    else:
        try:
            return uploaded_file.getvalue().decode("utf-8")
        except Exception:
            return str(uploaded_file.getvalue())


def compute_topic_scores(docs, years=None, top_n_terms=50):
    """Computes topic importance using TF-IDF and trend weighting."""
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
    X = vect.fit_transform(docs)
    feature_names = np.array(vect.get_feature_names_out())

    tfidf_mean = np.asarray(X.mean(axis=0)).ravel()
    term_counts = np.asarray((X > 0).sum(axis=0)).ravel()

    # Trend component
    if years is not None and len(years) == len(docs):
        yr_arr = np.array(years).astype(float)
        if yr_arr.max() == yr_arr.min():
            recent_weight = np.ones_like(yr_arr)
        else:
            recent_weight = (yr_arr - yr_arr.min()) / (yr_arr.max() - yr_arr.min())
        presence = (X > 0).astype(int).toarray()
        term_year_score = presence.T.dot(recent_weight)
        if term_year_score.max() != 0:
            term_year_score = term_year_score / term_year_score.max()
    else:
        term_year_score = np.ones_like(tfidf_mean)

    freq_norm = term_counts / (term_counts.max() if term_counts.max() > 0 else 1)
    composite = (
        0.5 * (tfidf_mean / (tfidf_mean.max() if tfidf_mean.max() > 0 else 1))
        + 0.3 * freq_norm
        + 0.2 * term_year_score
    )

    top_idx = np.argsort(composite)[::-1][:top_n_terms]
    results = pd.DataFrame({
        "term": feature_names[top_idx],
        "composite_score": composite[top_idx],
        "tfidf": tfidf_mean[top_idx],
        "doc_frequency": term_counts[top_idx],
        "trend_score": term_year_score[top_idx],
    })
    return results, vect


# ------------------ Streamlit UI ------------------
st.title("🧠 StudySage AI — Smart Study Companion")
st.markdown(
    """
Welcome to **StudySage AI**, your intelligent exam preparation assistant.
Upload your **past question papers (PDF/TXT)** and optionally your **syllabus**.
The app will analyze them using **Natural Language Processing (TF-IDF)** and
predict the **most important topics** likely to appear in upcoming exams.
"""
)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Upload & Settings")
    uploaded_files = st.file_uploader(
        "Upload past question papers (PDF/TXT)", accept_multiple_files=True
    )
    syllabus_file = st.file_uploader("Upload syllabus (optional)")
    use_years = st.checkbox("Assign years to each file", value=True)
    top_n = st.slider("Number of top topics to show", 5, 50, 10)
    use_nmf = st.checkbox("Show advanced topic modeling (NMF)", value=False)

# Default Demo Data
if not uploaded_files:
    st.info("No uploads detected — using demo data for preview.")
    docs = [
        "Machine learning basics classification regression supervised learning model evaluation accuracy precision recall",
        "Regression models linear regression multivariate regression error metrics mse rmse r2",
        "Neural networks perceptron backpropagation layers activation functions cnn rnn deep learning applications",
        "Feature engineering scaling normalization encoding missing values categorical features one hot encoding data preprocessing",
    ]
    years = [2018, 2019, 2021, 2024]
else:
    docs, years = [], []
    for f in uploaded_files:
        text = extract_text_from_upload(f)
        docs.append(text)
        if use_years:
            m = re.search(r"(19|20)\d{2}", f.name)
            years.append(int(m.group()) if m else None)
        else:
            years.append(None)

    # Ask user for manual mapping if year not detected
    if use_years and any(y is None for y in years):
        st.subheader("Map filenames to years")
        default_map = "\n".join([f"{f.name}: {2020+i}" for i, f in enumerate(uploaded_files)])
        mapping_text = st.text_area(
            "Enter filename:year (one per line)", value=default_map, height=120
        )
        name2year = {}
        for line in mapping_text.splitlines():
            if ":" in line:
                name, yr = line.split(":", 1)
                try:
                    name2year[name.strip()] = int(re.sub("[^0-9]", "", yr))
                except:
                    name2year[name.strip()] = 2020
        years = [name2year.get(f.name, years[i] if years[i] else 2020) for i, f in enumerate(uploaded_files)]

# Load syllabus if provided
syll_text = ""
if syllabus_file:
    sy_text = extract_text_from_upload(syllabus_file)
    syll_text = clean_text(sy_text)

# ------------------ Analysis Section ------------------
if st.button("🔍 Analyze Now"):
    if not docs:
        st.error("Please upload at least one document.")
    else:
        with st.spinner("Processing..."):
            cleaned_docs = [clean_text(d) for d in docs]
            if syll_text:
                cleaned_docs_with_syllabus = cleaned_docs + [syll_text]
                years_for_scoring = years + [max(years) + 1 if years else None]
            else:
                cleaned_docs_with_syllabus = cleaned_docs
                years_for_scoring = years if any(y is not None for y in years) else None

            results, vect = compute_topic_scores(cleaned_docs_with_syllabus, years=years_for_scoring, top_n_terms=200)

            if syll_text:
                results["in_syllabus"] = results["term"].apply(lambda t: 1 if t in syll_text else 0)
                results["final_score"] = results["composite_score"] + 0.12 * results["in_syllabus"]
            else:
                results["final_score"] = results["composite_score"]

            results = results.sort_values("final_score", ascending=False).reset_index(drop=True)

            # Display results
            st.success("Analysis Complete ✅")
            st.subheader("📊 Top Predicted Topics")
            display_df = results[["term", "final_score", "tfidf", "doc_frequency", "trend_score"]].head(top_n)
            display_df.columns = ["Topic", "Score", "Avg TF-IDF", "Doc Frequency", "Trend"]
            st.dataframe(display_df)

            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            display_df.plot(kind="barh", x="Topic", y="Score", ax=ax, legend=False)
            ax.set_xlabel("Predicted Importance Score")
            st.pyplot(fig)

            # Download CSV
            st.download_button(
                "⬇️ Download Topics CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="predicted_topics.csv",
                mime="text/csv",
            )

            # Text output
            st.subheader("📝 Top Topics Summary")
            st.text_area("Copy-ready topic list", "\n".join(results["term"].head(top_n)), height=200)

            # Optional Topic Modeling
            if use_nmf:
                st.subheader("🧩 NMF Topic Modeling (Experimental)")
                nmf_vect = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
                Xnmf = nmf_vect.fit_transform(cleaned_docs)
                n_topics = st.slider("Number of NMF topics", 2, 10, 5)
                nmf = NMF(n_components=n_topics, random_state=42)
                H = nmf.fit_transform(Xnmf)
                feature_names = nmf_vect.get_feature_names_out()
                for topic_idx, topic in enumerate(nmf.components_):
                    top_features = [feature_names[i] for i in topic.argsort()[:-11:-1]]
                    st.write(f"**Topic {topic_idx + 1}:** " + ", ".join(top_features))

st.markdown("---")
st.caption("© 2025 StudySage AI — Created by ANURAG SAINI THE BAKU")
