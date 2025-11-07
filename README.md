# 🧠 StudySage AI — Smart Study Companion

**StudySage AI** is an intelligent study assistant built by **ANURAG SAINI THE BAKU**.  
It uses **Natural Language Processing (NLP)** and **Machine Learning** to analyze past-year question papers and **predict the most important exam topics**.

🎓 **Official Live App:**  
🔗 [https://studysage-ai-zghwvw27phgne94ludtnhe.streamlit.app/](https://studysage-ai-zghwvw27phgne94ludtnhe.streamlit.app/)

---

## 🚀 **Project Overview**

StudySage AI helps students **learn smarter, not harder** by identifying the topics that are **most likely to appear** in future exams — based on trends and frequency patterns found in past papers.

### ✨ **Key Features**
- 📄 Upload **past question papers** (PDF or TXT)
- 📘 Optionally upload **syllabus** for context-based topic filtering
- 🧠 Analyze documents using **TF-IDF + trend weighting**
- 🔮 Predict and rank **important topics** by recurrence and significance
- 📊 Visualize top topics with dynamic bar charts
- ⬇️ Download results as CSV for future revision
- 🧩 (Optional) Explore advanced **NMF topic modeling**

---

## 🧩 **How It Works**

1. **Upload Files**  
   Add your past question papers (PDF or TXT). Optionally, include your syllabus.

2. **Processing**  
   The AI cleans and tokenizes the text, removing stopwords and noise.

3. **Topic Analysis**  
   Using **TF-IDF** and **trend scoring**, it ranks the most frequent and contextually important terms.

4. **Prediction Output**  
   Displays the most likely topics for upcoming exams, both in table and graph format.

5. **Download & Revise**  
   Export your results as CSV or copy them directly for your study notes.

---

## 🧠 **Technologies Used**

| Category | Technology |
|-----------|-------------|
| Programming | Python |
| Frontend | Streamlit |
| NLP | TF-IDF (Scikit-learn) |
| ML | Trend Weighting + Topic Modeling (NMF) |
| Visualization | Matplotlib |
| Document Parsing | PyPDF2 |

---

## 🧪 **Installation (Run Locally)**

1. Clone this repository or download the project zip.
2. Open the folder in your terminal or VS Code.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
