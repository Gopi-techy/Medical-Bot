from flask import Flask, render_template, jsonify, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage 
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for Flask sessions; change to a random string

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

# Create the conversational chain (memory handled per-session)
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=chatModel,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt}
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg
    print(input_text)
    
    # Load chat_history from session (deserialize)
    chat_history = [HumanMessage(**msg) if msg['type'] == 'human' else AIMessage(**msg) for msg in session.get('chat_history', [])]
    
    # Invoke the chain with chat_history
    response = rag_chain.invoke({"question": input_text, "chat_history": chat_history})
    answer = response["answer"]
    print("Response : ", answer)
    
    # Update chat_history
    chat_history.append(HumanMessage(content=input_text))
    chat_history.append(AIMessage(content=answer))
    
    # Save chat_history to session (serialize)
    session['chat_history'] = [msg.dict() for msg in chat_history]
    
    return str(answer)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050, debug=True)
