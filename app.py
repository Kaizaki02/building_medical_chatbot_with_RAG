from flask import Flask,render_template,request,jsonify
from  src.helpers import download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAI
from pinecone import Pinecone
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
import os
from src.prompt import *



app = Flask(__name__)

load_dotenv()


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') 

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


embeddings=download_hugging_face_embedding()

index_name = 'medical-chatbot'
#embed each chunk and upsert the embeddings into our pinecone index
docsearch = PineconeVectorStore(
    index=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type='similarity',search_kwargs={"k":3})
chatmodel = GoogleGenerativeAI(model_name="gemini-pro",temperature=0,max_output_tokens=1024)
prompt = ChatPromptTemplate.from_messages([
   {"system":system_prompt},
   {"human":"{input}"}
])


question_answer_chain = create_stuff_documents_chain(chatmodel,prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)






@app.route('/')
def index():
  return render_template('chat.html')


@app.route("/get",method=["GET","POST"])
def chat():
   msg = request.form["msg"]
   input = msg
   print(input)
   response = rag_chain.invoke({"input":msg})
   print("Response:",response['answer'])
   return str(response["answer"])

if __name__ == "__main__":
   app.run(host="0.0.0.0",port = 8080,debug = True)