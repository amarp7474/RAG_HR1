
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_aws import BedrockLLM
 
def hr_index():
    
    data_load=PyPDFLoader('https://www.insurance.ca.gov/01-consumers/105-type/95-guides/07-life/upload/LifeInsuranceAndAnnuities-2.pdf')  
    
    data_split=RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100,chunk_overlap=10)
    
    data_embeddings=BedrockEmbeddings(
    credentials_profile_name= 'default',
    model_id='amazon.titan-embed-text-v1')
    
    data_index=VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS)
    
    db_index=data_index.from_loaders([data_load])
    return db_index

def hr_llm():
    llm=BedrockLLM(
        credentials_profile_name='default',
        model_id='anthropic.claude-v2',
        model_kwargs={
        "max_tokens_to_sample":3000,
        "temperature": 0.1,
        "top_p": 0.9})
    return llm

def hr_rag_response(index,question):
    rag_llm=hr_llm()
    hr_rag_query=index.query(question=question,llm=rag_llm)
    return hr_rag_query
#  https://api.python.langchain.com/en/latest/indexes/langchain.indexes.vectorstore.VectorstoreIndexCreator.html