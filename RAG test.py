#1. Import OS, Document Loader, Text Splitter, Bedrock Embeddings, Vector DB, VectorStoreIndex, Bedrock-LLM
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.text_splitters import RecursiveCharacterTextSplitter

#2. Define the data source and load data with PDFLoader(https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf)
data_load=PyPDFLoader("https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf")

#3. Split the Text based on Character, Tokens etc. - Recursively split by character - ["\n\n", "\n", " ", ""]
data_split=RecursiveCharecterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=10)
data_sample= 'The sender of this email is registered with Naukri.com as Tvaksa Technologies. The responsibility of checking the authenticity of offers/correspondence lies with you. If you consider the content of this email inappropriate or spam, forward this email to compliance@naukri.com. You can also block this recruiter from searching your resume.'
data_test=data_split.split_text(data_sample)
print(data_test)

#4. Create Embeddings -- Client connection
#5à Create Vector DB, Store Embeddings and Index for Search - VectorstoreIndexCreator
#5b Create index for HR Report
#5c. Wrap within a function
#6a. Write a function to connect to Bedrock Foundation Model
#6b. Write a function which searches the user prompt, searches the best match from Vector DB and sends both to LLM.
# Index creation --> https://api.python.langchain.com/en/latest/indexes/langchain.indexes.vectorstore.VectorstoreIndexCreator.html