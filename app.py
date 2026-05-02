from flask import Flask, request, render_template, jsonify, session
import pickle, re, os
import PyPDF2
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = "resumescreener2024"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model   = pickle.load(open("model.pkl",   "rb"))
tfidf   = pickle.load(open("tfidf.pkl",   "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_text_from_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_explanation(resume_text, predicted_role, broad_role):
    try:
        prompt = f"""A resume screener predicted the role: "{predicted_role}" (category: {broad_role}).
Resume text: {resume_text[:2000]}
In 3-4 sentences explain WHY. Mention actual skills found. Be friendly. Also mention overlap with other roles if any."""
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate explanation: {str(e)}"

def get_specific_role(broad_role, resume_text):
    text = resume_text.lower()

    def has(keywords):
        return sum(1 for k in keywords if k in text)

    if broad_role == "INFORMATION-TECHNOLOGY":
        scores = {
            "Full Stack Developer":   has(["react","angular","vue","nodejs","node","express","mongodb","fullstack","full stack","frontend","backend","html","css","javascript"]),
            "Frontend Developer":     has(["react","angular","vue","html","css","javascript","typescript","ui","ux","figma","bootstrap","responsive"]),
            "Backend Developer":      has(["nodejs","node","express","django","flask","spring","rest api","microservices","postgresql","mysql","mongodb"]),
            "Python Developer":       has(["python","django","flask","fastapi","pandas","numpy","scripting"]),
            "Java Developer":         has(["java","spring","springboot","hibernate","maven","gradle","j2ee","jsp"]),
            "Android Developer":      has(["android","kotlin","java","mobile","playstore","sdk","xml layout"]),
            "iOS Developer":          has(["ios","swift","xcode","objective c","cocoa","appstore"]),
            "ML Engineer":            has(["machine learning","deep learning","tensorflow","pytorch","scikit","model","nlp","neural"]),
            "Data Scientist":         has(["data science","statistics","python","r language","pandas","numpy","visualization","hypothesis","regression"]),
            "Data Analyst":           has(["sql","excel","tableau","power bi","dashboard","analytics","reporting","pivot"]),
            "DevOps Engineer":        has(["docker","kubernetes","jenkins","ci cd","terraform","ansible","aws","azure","gcp","linux","bash"]),
            "Cloud Engineer":         has(["aws","azure","gcp","cloud","s3","ec2","lambda","vpc","cloudformation"]),
            "Network Engineer":       has(["networking","cisco","router","switch","firewall","tcp ip","vpn","network"]),
            "Cybersecurity Analyst":  has(["security","penetration","ethical hacking","firewall","siem","vulnerability","encryption"]),
            "Database Administrator": has(["sql","mysql","postgresql","oracle","database","dba","backup","query","nosql"]),
            "Software Developer":     has(["software","development","programming","agile","scrum","git","github","api"]),
        }
        return max(scores, key=scores.get)

    elif broad_role == "ENGINEERING":
        scores = {
            "AI Engineer":            has(["artificial intelligence","machine learning","deep learning","neural","tensorflow","pytorch"]),
            "DevOps Engineer":        has(["docker","kubernetes","jenkins","ci cd","terraform","ansible","devops"]),
            "Mechanical Engineer":    has(["mechanical","autocad","solidworks","manufacturing","thermodynamics","production"]),
            "Civil Engineer":         has(["civil","construction","autocad","structural","surveying","concrete","roads"]),
            "Electrical Engineer":    has(["electrical","circuit","plc","scada","power","electronics","embedded"]),
            "Chemical Engineer":      has(["chemical","process","refinery","reaction","thermodynamics","plant"]),
            "Software Engineer":      has(["software","programming","development","algorithms","data structures","git"]),
        }
        return max(scores, key=scores.get)

    elif broad_role == "DESIGNER":
        scores = {
            "UI/UX Designer":         has(["figma","adobe xd","wireframe","prototype","user experience","ui","ux"]),
            "Graphic Designer":       has(["photoshop","illustrator","corel","branding","logo","typography","print"]),
            "Web Designer":           has(["html","css","javascript","responsive","bootstrap","web design"]),
        }
        return max(scores, key=scores.get)

    elif broad_role == "DIGITAL-MEDIA":
        scores = {
            "Social Media Manager":   has(["instagram","facebook","twitter","content","engagement","social media"]),
            "SEO Specialist":         has(["seo","google analytics","keywords","ranking","backlinks","search engine"]),
            "Content Writer":         has(["writing","blogging","content","copywriting","articles","seo writing"]),
            "Video Editor":           has(["video","premiere","after effects","editing","youtube","motion"]),
        }
        return max(scores, key=scores.get)

    elif broad_role == "BUSINESS-DEVELOPMENT":
        scores = {
            "Business Analyst":       has(["business analysis","requirements","stakeholder","process","documentation","brd"]),
            "Product Manager":        has(["product","roadmap","agile","scrum","sprint","user stories","kpi"]),
            "Sales Manager":          has(["sales","revenue","crm","leads","targets","b2b","b2c","pipeline"]),
            "Marketing Manager":      has(["marketing","campaign","brand","digital marketing","seo","ads","strategy"]),
        }
        return max(scores, key=scores.get)

    role_names = {
        "ACCOUNTANT": "Accountant", "ADVOCATE": "Advocate / Lawyer",
        "AGRICULTURE": "Agriculture Specialist", "APPAREL": "Apparel / Fashion",
        "ARTS": "Artist / Creative", "AUTOMOBILE": "Automobile Engineer",
        "AVIATION": "Aviation Professional", "BANKING": "Banking Professional",
        "BPO": "BPO / Customer Support", "CHEF": "Chef / Culinary",
        "CONSTRUCTION": "Construction Manager", "CONSULTANT": "Consultant",
        "FINANCE": "Finance Analyst", "FITNESS": "Fitness Trainer",
        "HEALTHCARE": "Healthcare Professional", "HR": "HR Manager",
        "PUBLIC-RELATIONS": "Public Relations", "SALES": "Sales Executive",
        "TEACHER": "Teacher / Educator"
    }
    return role_names.get(broad_role, broad_role)

ROLE_ICONS = {
    "Full Stack Developer": "🌐", "Frontend Developer": "🖥️",
    "Backend Developer": "⚙️", "Python Developer": "🐍",
    "Java Developer": "☕", "Android Developer": "📱",
    "iOS Developer": "🍎", "ML Engineer": "🤖",
    "Data Scientist": "📊", "Data Analyst": "📉",
    "DevOps Engineer": "🚀", "Cloud Engineer": "☁️",
    "Network Engineer": "🔌", "Cybersecurity Analyst": "🔒",
    "Database Administrator": "🗄️", "Software Developer": "💻",
    "Software Engineer": "🛠️", "AI Engineer": "🧠",
    "Mechanical Engineer": "🔧", "Civil Engineer": "🏗️",
    "Electrical Engineer": "⚡", "Chemical Engineer": "🧪",
    "UI/UX Designer": "🎨", "Graphic Designer": "✏️",
    "Web Designer": "🖌️", "Social Media Manager": "📱",
    "SEO Specialist": "🔍", "Content Writer": "✍️",
    "Video Editor": "🎬", "Business Analyst": "📈",
    "Product Manager": "📋", "Sales Manager": "💼",
    "Marketing Manager": "📣", "Accountant": "🧾",
    "Advocate / Lawyer": "⚖️", "Agriculture Specialist": "🌾",
    "Apparel / Fashion": "👗", "Artist / Creative": "🎭",
    "Automobile Engineer": "🚗", "Aviation Professional": "✈️",
    "Banking Professional": "🏦", "BPO / Customer Support": "📞",
    "Chef / Culinary": "👨‍🍳", "Construction Manager": "🏗️",
    "Consultant": "💼", "Finance Analyst": "💰",
    "Fitness Trainer": "💪", "Healthcare Professional": "🏥",
    "HR Manager": "👥", "Public Relations": "📢",
    "Sales Executive": "🛒", "Teacher / Educator": "👨‍🏫"
}

@app.route("/")
def index():
    return render_template("index.html", prediction=None, icon=None, broad=None, explanation=None)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("resume")
    if not file or file.filename == "":
        return render_template("index.html", prediction="❌ Please upload a PDF file.", icon="❌", broad=None, explanation=None)
    if not file.filename.endswith(".pdf"):
        return render_template("index.html", prediction="❌ Only PDF files are supported.", icon="❌", broad=None, explanation=None)

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    raw_text = extract_text_from_pdf(path)

    if not raw_text.strip():
        return render_template("index.html", prediction="❌ Could not read text from PDF.", icon="❌", broad=None, explanation=None)

    cleaned    = clean_text(raw_text)
    features   = tfidf.transform([cleaned])
    pred_label = model.predict(features)[0]
    broad_role = encoder.inverse_transform([pred_label])[0]
    specific   = get_specific_role(broad_role, raw_text)
    icon       = ROLE_ICONS.get(specific, "💼")

    explanation = generate_explanation(raw_text, specific, broad_role)

    session["resume_text"]    = raw_text[:3000]
    session["predicted_role"] = specific

    return render_template("index.html", prediction=specific, icon=icon, broad=broad_role, explanation=explanation)

@app.route("/chat", methods=["POST"])
def chat():
    data           = request.json
    user_question  = data.get("question", "")
    resume_text    = session.get("resume_text", "")
    predicted_role = session.get("predicted_role", "Unknown")

    if not user_question:
        return jsonify({"reply": "Please ask a question."})

    try:
        prompt = f"""You are a friendly AI career advisor. The user uploaded their resume and got predicted role: "{predicted_role}".
Resume snippet: {resume_text[:1500]}
User question: {user_question}
Answer helpfully in 2-4 sentences. Be specific about their resume skills."""
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"reply": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)