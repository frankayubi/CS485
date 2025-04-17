from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from src.question_tracker import QuestionTracker
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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

if __name__ == '__main__':
    # Run this for local development
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)