import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import nltk, re, os
from PyPDF2 import PdfReader
import joblib

# NLTK setup (Streamlit cloud will run pip install and nltk download in environment)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('english'))

st.set_page_config(page_title='Smart Study Companion', layout='wide')

def read_pdf(file) -> str:
    try:
        reader = PdfReader(file)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return \"\\n\".join(text)
    except Exception:
        return \"\"

def clean_text(text: str) -> str:
    if text is None:
        return ''
    text = text.replace('\\n', ' ').lower()
    text = re.sub(r'[^a-z0-9\\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2 and not t.isdigit()]
    return ' '.join(tokens)

def extract_text_from_upload(uploaded_file):
    fname = uploaded_file.name.lower()
    if fname.endswith('.pdf'):
        return read_pdf(uploaded_file)
    else:
        try:
            return uploaded_file.getvalue().decode('utf-8')
        except Exception:
            return str(uploaded_file.getvalue())

def compute_topic_scores(docs, years=None, top_n_terms=50):
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
    X = vect.fit_transform(docs)
    feature_names = np.array(vect.get_feature_names_out())
    tfidf_mean = np.asarray(X.mean(axis=0)).ravel()
    term_counts = np.asarray((X > 0).sum(axis=0)).ravel()
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
    composite = 0.5 * (tfidf_mean / (tfidf_mean.max() if tfidf_mean.max() > 0 else 1)) + \
                0.3 * freq_norm + \
                0.2 * term_year_score
    top_idx = np.argsort(composite)[::-1][:top_n_terms]
    results = pd.DataFrame({
        'term': feature_names[top_idx],
        'composite_score': composite[top_idx],
        'tfidf': tfidf_mean[top_idx],
        'doc_frequency': term_counts[top_idx],
        'trend_score': term_year_score[top_idx]
    })
    return results, vect

st.title('📚 Smart Study Companion — Streamlit')
st.markdown('Upload past question papers (PDF/TXT) and an optional syllabus. The app will predict most likely topics using TF-IDF + trend weighting.')

with st.sidebar:
    st.header('Upload & Settings')
    uploaded_files = st.file_uploader('Upload past question papers (PDF or TXT) — multiple', accept_multiple_files=True)
    syllabus_file = st.file_uploader('Upload syllabus (optional)', help='Plain text or PDF syllabus helps filter topics')
    use_years = st.checkbox('Assign year labels to each uploaded paper (recommended for trends)', value=True)
    top_n = st.slider('Number of top topics to show', min_value=5, max_value=50, value=12)
    use_nmf = st.checkbox('Show NMF topic modelling (experimental)', value=False)

# Example mode fallback
if not uploaded_files:
    st.info('No uploads detected — using demo documents. Replace with your own for meaningful results.')
    docs = [
        'machine learning basics classification regression supervised learning model evaluation accuracy precision recall',
        'regression models linear regression multivariate regression error metrics mse rmse r2',
        'neural networks perceptron backpropagation layers activation functions cnn rnn',
        'feature engineering scaling normalization encoding missing values categorical features one hot encoding'
    ]
    years = [2018, 2019, 2021, 2024]
else:
    docs = []
    years = []
    for f in uploaded_files:
        text = extract_text_from_upload(f)
        docs.append(text)
        if use_years:
            # try to parse year from filename; if not found user mapping later
            m = re.search(r'(19|20)\\d{2}', f.name)
            years.append(int(m.group()) if m else None)
        else:
            years.append(None)
    # If any year None and use_years, ask user mapping
    if use_years and any(y is None for y in years):
        st.subheader('Map uploaded filenames to years (one per line, filename:year)')
        default_map = '\\n'.join([f'{f.name}: {2020+i}' for i,f in enumerate(uploaded_files)])
        mapping_text = st.text_area('Filename:Year mappings', value=default_map, height=140)
        name2year = {}
        for line in mapping_text.splitlines():
            if ':' in line:
                name, yr = line.split(':',1)
                try:
                    name2year[name.strip()] = int(re.sub('[^0-9]','', yr))
                except:
                    name2year[name.strip()] = 2020
        years = [name2year.get(f.name, years[i] if years[i] is not None else 2020) for i,f in enumerate(uploaded_files)]

syll_text = ''
if syllabus_file:
    sy_text = extract_text_from_upload(syllabus_file)
    syll_text = clean_text(sy_text)

if st.button('Analyze'):
    if not docs:
        st.error('No documents loaded. Upload papers or use demo mode.')
    else:
        with st.spinner('Processing...'):
            cleaned_docs = [clean_text(d) for d in docs]
            if syll_text:
                cleaned_docs_with_syllabus = cleaned_docs + [syll_text]
                years_for_scoring = years + [max(years)+1 if years else None]
            else:
                cleaned_docs_with_syllabus = cleaned_docs
                years_for_scoring = years if any(y is not None for y in years) else None
            results, vect = compute_topic_scores(cleaned_docs_with_syllabus, years=years_for_scoring, top_n_terms=200)
            if syll_text:
                results['in_syllabus'] = results['term'].apply(lambda t: 1 if t in syll_text else 0)
                results['final_score'] = results['composite_score'] + 0.12 * results['in_syllabus']
            else:
                results['final_score'] = results['composite_score']
            results = results.sort_values('final_score', ascending=False).reset_index(drop=True)
            st.subheader('Top Predicted Topics')
            display_df = results[['term','final_score','tfidf','doc_frequency','trend_score']].head(top_n)
            display_df = display_df.rename(columns={'term':'Topic','final_score':'Score','tfidf':'Avg TF-IDF','doc_frequency':'Doc Freq','trend_score':'Trend'})
            st.dataframe(display_df.style.format({'Score':'{:.3f}','Avg TF-IDF':'{:.4f}','Doc Freq':'{:.0f}','Trend':'{:.3f}'}))
            fig, ax = plt.subplots(figsize=(10,4))
            top = display_df[['Topic','Score']].set_index('Topic').sort_values('Score')
            top.plot(kind='barh', legend=False, ax=ax)
            ax.set_xlabel('Predicted Importance (composite score)')
            st.pyplot(fig)
            st.download_button('Download Topics CSV', data=results[['term','final_score','tfidf','doc_frequency','trend_score']].to_csv(index=False).encode('utf-8'), file_name='predicted_topics.csv', mime='text/csv')
            st.subheader('Copy / Paste - Top Topics')
            top_text = '\\n'.join([f\"{i+1}. {t}\" for i,t in enumerate(results['term'].head(top_n))])
            st.text_area('Top Topics', value=top_text, height=200)
            if use_nmf:
                st.subheader('NMF Topics (experimental)')
                nmf_vect = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
                Xnmf = nmf_vect.fit_transform(cleaned_docs)
                n_topics = st.slider('Number of NMF topics', 2, 12, 5)
                from sklearn.decomposition import NMF
                nmf = NMF(n_components=n_topics, random_state=42)
                W = nmf.fit_transform(Xnmf)
                H = nmf.components_
                feature_names = nmf_vect.get_feature_names_out()
                for topic_idx, topic in enumerate(H):
                    top_features_ind = topic.argsort()[:-11:-1]
                    top_features = [feature_names[i] for i in top_features_ind]
                    st.write(f'**Topic {topic_idx}**: ' + ', '.join(top_features))
            st.success('Analysis complete! Use the CSV or the copy/paste box to prepare your study plan.')
st.markdown('---')
st.caption('Smart Study Companion — made for ANURAG SAINI THE BAKU')