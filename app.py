from flask import Flask, render_template, request, jsonify
from src.helpers import download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt

app = Flask(__name__)

# .env must contain PINECONE_API_KEY and GROQ_API_KEY
load_dotenv(override=True)

embeddings = download_hugging_face_embedding()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
chatmodel = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chatmodel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    msg = request.form.get("query", "").strip()
    if not msg:
        return jsonify(reply="Please type a question."), 400
    response = rag_chain.invoke({"input": msg})
    return jsonify(reply=response["answer"])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)