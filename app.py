from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from src.question_tracker import QuestionTracker
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
import json
import os
from langchain.document_loaders import PyPDFLoader
from src.helper import text_split

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Configure upload settings
UPLOAD_FOLDER = 'Data'
ALLOWED_EXTENSIONS = {'pdf'}

# Add to your app configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')

# Simple admin credentials - in production use a proper database
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = generate_password_hash('your-secure-password')

embeddings = download_hugging_face_embeddings()
tracker = QuestionTracker()  # Initialize the question tracker

index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = OpenAI(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_new_pdf(file_path):
    """Process a newly uploaded PDF and add it to the Pinecone index"""
    # Load the single PDF file
    loader = PyPDFLoader(file_path)
    extracted_data = loader.load()
    
    # Split into chunks
    text_chunks = text_split(extracted_data)
    
    # Connect to existing index
    index_name = "medicalbot"
    
    # Add new documents to the existing index
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    # Add the new chunks to the index
    docsearch.add_documents(text_chunks)
    
    return len(text_chunks)

@app.route("/")
def index():
    # Get the most common user questions for the quick questions feature
    common_questions = tracker.get_most_common()
    return render_template('chat.html', common_questions=common_questions)

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    
    # Track this question for future common questions
    tracker.add_question(msg)
    
    print(f"Processing query: {msg}")
    response = rag_chain.invoke({"input": msg})
    print(f"Response: {response['answer']}")
    
    return str(response["answer"])

@app.route("/refresh_questions", methods=["GET"])
def refresh_questions():
    """Endpoint to refresh the common questions without reloading the page"""
    common_questions = tracker.get_most_common()
    return jsonify({"questions": common_questions})

@app.route('/admin')
def admin():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # Get list of PDFs
    pdf_files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.endswith('.pdf'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_stats = os.stat(file_path)
            
            pdf_files.append({
                'filename': filename,
                'size': f"{file_stats.st_size / 1024:.1f} KB",
                'upload_date': datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M')
            })
    
    # Get the number of questions
    with open('user_questions.json', 'r') as f:
        questions_data = json.load(f)
        question_count = len(questions_data.get('questions', []))
    
    # Get the most recent upload date
    last_upload = "No uploads yet"
    if pdf_files:
        last_upload = max(pdf_files, key=lambda x: x['upload_date'])['upload_date']
    
    return render_template(
        'admin.html', 
        pdfs=pdf_files, 
        pdf_count=len(pdf_files), 
        question_count=question_count,
        last_upload=last_upload
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD, password):
            session['logged_in'] = True
            return redirect(url_for('admin'))
        else:
            flash('Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('index'))

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('admin'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('admin'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the uploaded PDF
        chunks_count = process_new_pdf(file_path)
        
        flash(f'File successfully uploaded and processed. Added {chunks_count} chunks to the index.')
        return redirect(url_for('admin'))
    
    return redirect(url_for('admin'))

if __name__ == '__main__':
    # Run this for local development
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)