# ============================================================
# 🧠 StudySage AI — Smart Study Companion (NLTK-free tokenization)
# Author: ANURAG SAINI THE BAKU (adapted)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from PyPDF2 import PdfReader

st.set_page_config(page_title="StudySage AI", layout="wide")

# ------------------ Lightweight stopwords (small reliable set) ------------------
STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","as","at",
    "be","because","been","before","being","below","between","both","but","by",
    "could","did","do","does","doing","down","during",
    "each","few","for","from","further",
    "had","has","have","having","he","her","here","hers","herself","him","himself","his","how",
    "i","if","in","into","is","it","its","itself",
    "just",
    "me","more","most","my","myself",
    "no","nor","not","now",
    "of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own",
    "same","she","should","so","some","such",
    "than","that","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too",
    "under","until","up",
    "very",
    "was","we","were","what","when","where","which","while","who","whom","why","with","would",
    "you","your","yours","yourself","yourselves"
}

# ------------------ Helper functions ------------------

def read_pdf(file) -> str:
    """Extract text from a PDF file object or path-like."""
    try:
        reader = PdfReader(file)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_from_upload(uploaded_file):
    """Return text for uploaded file (PDF or TXT)"""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    else:
        try:
            return uploaded_file.getvalue().decode("utf-8")
        except Exception:
            return str(uploaded_file.getvalue())

def simple_tokenize(text):
    """Tokenize with regex, return lowercase alpha tokens length>=3."""
    if text is None:
        return []
    # keep only letters and numbers and spaces; replace others with space
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = text.lower()
    tokens = re.findall(r"\b[a-z]{3,}\b", text)  # words with only letters and length>=3
    return tokens

def clean_text(text):
    """Clean text and remove stopwords using regex-based tokenization."""
    tokens = simple_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def compute_topic_scores(docs, years=None, top_n_terms=50):
    """Compute TF-IDF based topic ranking with optional trend weighting."""
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
    X = vect.fit_transform(docs)
    feature_names = np.array(vect.get_feature_names_out())

    tfidf_mean = np.asarray(X.mean(axis=0)).ravel()
    term_counts = np.asarray((X > 0).sum(axis=0)).ravel()

    # trend score
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
Upload past question papers (PDF/TXT) and an optional syllabus.
StudySage AI analyzes historical papers using TF-IDF + trend-weighting
to predict which topics are most likely to appear in future exams.
"""
)

with st.sidebar:
    st.header("⚙️ Upload & Settings")
    uploaded_files = st.file_uploader("Upload past question papers (PDF/TXT)", accept_multiple_files=True)
    syllabus_file = st.file_uploader("Upload syllabus (optional)")
    use_years = st.checkbox("Assign years to each file (detect from filename)", value=True)
    top_n = st.slider("Number of top topics to show", min_value=5, max_value=50, value=10)
    use_nmf = st.checkbox("Show NMF topic model (experimental)", value=False)

# Fallback demo docs
if not uploaded_files:
    st.info("No files uploaded — showing demo documents. Upload your own papers for real results.")
    docs = [
        "machine learning basics classification regression supervised learning model evaluation accuracy precision recall",
        "regression models linear regression multivariate regression error metrics mse rmse r2",
        "neural networks perceptron backpropagation cnn rnn deep learning applications",
        "feature engineering scaling normalization encoding missing values categorical features one hot encoding"
    ]
    years = [2018, 2019, 2021, 2024]
else:
    docs = []
    years = []
    filenames = []
    for f in uploaded_files:
        filenames.append(f.name)
        txt = extract_text_from_upload(f)
        docs.append(txt)
        if use_years:
            m = re.search(r"(19|20)\d{2}", f.name)
            years.append(int(m.group()) if m else None)
        else:
            years.append(None)

    # If years missing, let user map
    if use_years and any(y is None for y in years):
        st.subheader("Map filenames to years (filename:year)")
        default_map = "\n".join([f"{uploaded_files[i].name}: {2020+i}" for i in range(len(uploaded_files))])
        mapping_text = st.text_area("Enter mapping lines", value=default_map, height=160)
        name2year = {}
        for line in mapping_text.splitlines():
            if ":" in line:
                name, yr = line.split(":", 1)
                try:
                    name2year[name.strip()] = int(re.sub("[^0-9]", "", yr))
                except:
                    name2year[name.strip()] = 2020
        years = [name2year.get(uploaded_files[i].name, years[i] if years[i] else 2020) for i in range(len(uploaded_files))]

# Syllabus
syll_text = ""
if syllabus_file:
    syll_text_raw = extract_text_from_upload(syllabus_file)
    syll_text = clean_text(syll_text_raw)

if st.button("🔍 Analyze Now"):
    if not docs:
        st.error("Please upload at least one document.")
    else:
        with st.spinner("Analyzing..."):
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

            st.success("Analysis complete ✅")
            st.subheader("📊 Top Predicted Topics")
            display_df = results[["term", "final_score", "tfidf", "doc_frequency", "trend_score"]].head(top_n)
            display_df.columns = ["Topic", "Score", "Avg TF-IDF", "Doc Frequency", "Trend"]
            st.dataframe(display_df)

            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            display_df.plot(kind="barh", x="Topic", y="Score", ax=ax, legend=False)
            ax.set_xlabel("Predicted Importance Score")
            st.pyplot(fig)

            st.download_button(
                "⬇️ Download Topics CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="predicted_topics.csv",
                mime="text/csv",
            )

            st.subheader("📝 Copy-ready Topic List")
            st.text_area("Top topics", "\n".join(results["term"].head(top_n)), height=200)

            if use_nmf:
                st.subheader("🧩 NMF Topic Modeling (experimental)")
                nmf_vect = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
                Xnmf = nmf_vect.fit_transform(cleaned_docs)
                n_topics = st.slider("Number of NMF topics", 2, 10, 5)
                nmf = NMF(n_components=n_topics, random_state=42)
                nmf.fit(Xnmf)
                feature_names = nmf_vect.get_feature_names_out()
                for topic_idx, comp in enumerate(nmf.components_):
                    top_features = [feature_names[i] for i in comp.argsort()[:-11:-1]]
                    st.write(f"**Topic {topic_idx + 1}:** " + ", ".join(top_features))

st.markdown("---")
st.caption("© 2025 StudySage AI — Created by ANURAG SAINI THE BAKU")
