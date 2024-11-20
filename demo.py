import os
import json
import openai
import warnings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

# Supress stupid boring warings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize OpenAI API key
apikey = "add key here"
openai.api_key = apikey

def load_documents(folder_path):
    documents = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load_and_split(text_splitter))
        elif file_name.endswith(".json"):
            loader = JSONLoader(file_path)
            documents.extend(loader.load_and_split(text_splitter))
        elif file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load_and_split(text_splitter))

    return documents

def build_vector_store(documents):
    embeddings = OpenAIEmbeddings(openai_api_key=apikey) #embeddings to convert text documents into vectors for the chatbot to use.
    vector_store = FAISS.from_documents(documents, embeddings) #Facebook's similarity search. Pairs similar embeddings together for quicker access.
    return vector_store

def chat_with_rag(vector_store):
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=apikey)

    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("Chatbot initialized! Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = qa_chain.run(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    folder_path = "./rag_folder"

    documents = load_documents(folder_path)
    vector_store = build_vector_store(documents)

    chat_with_rag(vector_store)
