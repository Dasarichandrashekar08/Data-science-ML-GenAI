from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
import os


os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
sec_key = "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"

loader = DirectoryLoader('/content/new_articles',
                         glob="./*.txt", loader_cls=TextLoader)

document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
text = text_splitter.split_documents(document)


persist_directory = 'db'

# Create the HuggingFaceEmbeddings object
embedding = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")

# Create the Chroma vector store
vectordb = Chroma.from_documents(
    documents=text, embedding=embedding, persist_directory=persist_directory)

retriver = vectordb.as_retriever(search_kwargs={"k": 2})
retriver.search_type


repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128,
                          temperature=0.7, token=sec_key)

repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128,
                          temperature=0.7, token=sec_key)

llm.invoke("who chandra shekar")

retriever = vectordb.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Replace with the correct chain type
    retriever=retriever,
    return_source_documents=True
)


query = "how much money does microsoft raises"
llm_response1 = qa_chain(query)
print(llm_response1)
