import os
import string
import docx
import numpy as np
import PyPDF2
import nltk
import logging
from flask import Flask, render_template, request, redirect, url_for, session
from flask_caching import Cache
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inisialisasi Flask
app = Flask(__name__)
app.secret_key = "supersecretkey"
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {"pdf", "docx"}

# Pastikan dataset NLTK sudah tersedia
nltk.data.path.append("C:/Users/Agus Ikhsan/AppData/Roaming/nltk_data")
nltk.download('punkt')
nltk.download('stopwords')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi ekstraksi teks dari PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text.strip() if text else ""
    except Exception as e:
        logging.error(f"Error membaca PDF: {e}")
        return ""

# Fungsi ekstraksi teks dari DOCX
def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip() if text else ""
    except Exception as e:
        logging.error(f"Error membaca DOCX: {e}")
        return ""

# Fungsi preprocessing teks
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    return ' '.join([word for word in words if word not in stop_words])

# Fungsi menghitung kemiripan Cosine
def cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = np.dot(vectors[0], vectors[1]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
    return round(cosine_sim * 100, 2)

# Fungsi deteksi plagiarisme
def detect_plagiarism(uploaded_text, existing_texts):
    return max((cosine_similarity(uploaded_text, text) for text in existing_texts), default=0.0)

# Fungsi klasifikasi teks AI/Human
def classify_text(text):
    ai_keywords = {'neural network': 3, 'machine learning': 3, 'deep learning': 3, 'algorithm': 2}
    human_keywords = {'experience': 3, 'feeling': 3, 'thought': 3, 'emotion': 3}
    
    ai_score = sum(weight for keyword, weight in ai_keywords.items() if keyword in text.lower())
    human_score = sum(weight for keyword, weight in human_keywords.items() if keyword in text.lower())
    
    total_score = ai_score + human_score
    if total_score == 0:
        return 0  # No relevant keywords found
    return (ai_score / total_score) * 100

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            session.pop('similarity', None)
            session.pop('classification', None)
            return render_template("error.html", error="File tidak valid")  # Jika file tidak valid, arahkan ke error.html
        
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        
        uploaded_text = extract_text_from_pdf(path) if file.filename.endswith('.pdf') else extract_text_from_docx(path)
        os.remove(path)  # Hapus file setelah diproses
        
        if uploaded_text:
            existing_texts = [
                extract_text_from_pdf(os.path.join(UPLOAD_FOLDER, f)) if f.endswith('.pdf') else extract_text_from_docx(os.path.join(UPLOAD_FOLDER, f))
                for f in os.listdir(UPLOAD_FOLDER) if f != file.filename
            ]
            similarity_score = detect_plagiarism(uploaded_text, existing_texts)
            ai_percentage = classify_text(uploaded_text)
            
            # Simpan hasil dalam session agar tetap ada setelah refresh
            session['similarity'] = similarity_score
            session['classification'] = f"{ai_percentage:.2f}% AI-generated"
            session.modified = True  # Pastikan session diperbarui
        else:
            session.pop('similarity', None)
            session.pop('classification', None)
            return render_template("error.html", error="Tidak ada teks yang dapat diproses")
        
        return redirect(url_for("index"))
    
    return render_template('index.html', similarity=session.get('similarity', 0), classification=session.get('classification', 'N/A'))

@app.errorhandler(413)
def request_entity_too_large(_):
    return render_template("error.html", error="Ukuran file terlalu besar"), 413

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
