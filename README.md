# 🧠 AI Resume Screener (ML + Explainable AI)

An end-to-end **Machine Learning + AI-powered web application** that analyzes resumes and predicts the most suitable job role — along with a clear explanation of *why*.

🔗 **Live Demo**: https://resume-screener-mqz5.onrender.com/

---

## 🚀 Features

- 📄 Upload PDF Resume
- 🎯 Predict Job Role (20+ categories)
- 🧠 Smart Sub-role Detection (e.g., Full Stack, Data Analyst)
- 🤖 AI-generated Explanation (using LLM)
- 🌐 Deployed Web Application (Render)
- 🎨 Modern animated UI (Galaxy theme)

---

## 🧠 How It Works

### 1. Resume Processing
- Extracts text from PDF using **PyPDF2**
- Cleans text (removes symbols, links, noise)

### 2. Feature Engineering
- Uses **TF-IDF Vectorization**
- Converts resume text into numerical features

### 3. Machine Learning Model
- Trained on **~2100+ resumes dataset (CSV)**
- Predicts **broad job category**
- Model files:
  - `model.pkl`
  - `tfidf.pkl`
  - `encoder.pkl`

### 4. Smart Role Mapping
- Custom **keyword-based scoring system**
- Converts broad role → **specific role**

Example:

INFORMATION-TECHNOLOGY → Full Stack Developer


### 5. AI Explanation (Groq API)
- Uses **LLaMA 3.3 (via Groq API)**
- Generates:
  - Why this role was predicted
  - Skills detected in resume
  - Related career roles

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|----------|
| Backend | Flask (Python) |
| ML | Scikit-learn |
| NLP | TF-IDF |
| PDF Processing | PyPDF2 |
| AI | Groq API (LLaMA 3) |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render |
| Version Control | GitHub |

---

## 📂 Project Structure


resume-screener/
│── app.py
│── model.pkl
│── tfidf.pkl
│── encoder.pkl
│── requirements.txt
│── Procfile
│── templates/
│ └── index.html
│── uploads/


---

## 📊 Model Details

- Dataset: Resume CSV dataset (~2100+ resumes)
- Categories: 20+ job roles
- Accuracy: ~72–73%
- Model Type: TF-IDF + Linear ML classifier (SVM/Logistic)

---

## 🔐 Environment Variables

Create a `.env` file:


GROQ_API_KEY=your_api_key_here


---

## ▶️ Run Locally

```bash
git clone https://github.com/MMahimashree/resume-screener.git
cd resume-screener

pip install -r requirements.txt
python app.py

Open:

http://127.0.0.1:5000
🌐 Deployment

Deployed on Render:

Connected GitHub repository

Added Procfile:

web: gunicorn app:app
Configured environment variables (API key)
Automatic deployment on push
💡 Key Highlights
Combines ML + Rule-based logic + LLM
Provides Explainable AI output
Real-world application of NLP in hiring
Clean and interactive UI
🔮 Future Improvements
Confidence score display
Resume improvement suggestions
DOCX file support
Job matching integration


👩‍💻 Author
M Mahimashree
Computer Science Engineering Student
