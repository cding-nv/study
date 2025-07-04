import getpass
import os

# Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
#os.environ['FAISS_NO_AVX2'] = '1'

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
#from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


loader = TextLoader("./state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

#embeddings = OpenAIEmbeddings()
bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
#db = await FAISS.afrom_documents(docs, bge_embeddings)
db = FAISS.from_documents(docs, bge_embeddings)

query = "What did the president say about Ketanji Brown Jackson"

#docs = await db.asimilarity_search(query)
docs = db.similarity_search(query)

print(docs[0].page_content)
