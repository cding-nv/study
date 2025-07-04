
## Reference:
## https://cloud.tencent.com/developer/article/2373282
## https://github.com/netease-youdao/BCEmbedding
 
import os, dotenv
dotenv.load_dotenv()
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

# Load data
from langchain_community.document_loaders import TextLoader
loader = TextLoader('/home/sdp/fengding/RAG/doupocangqiong.txt', encoding='GB2312')
#loader = TextLoader('./xxx.txt', encoding='ascii')
documents = loader.load()

# Split data to chunks
#from langchain_community.text_splitter import CharacterTextSplitter
#text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
#chunks = text_splitter.split_documents(documents)

# Generate embedding for chunks and save together
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions

client = weaviate.Client(
  embedded_options = EmbeddedOptions()
)

bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
vectorstore = Weaviate.from_documents(
    client = client,
    documents = documents,
    #embedding = OpenAIEmbeddings(),
    embedding = bge_embeddings,
    by_text = False
)

# Retriever
retriever = vectorstore.as_retriever()

# Prompt Augment
from langchain.prompts import ChatPromptTemplate
template = """你是一个问答机器人助手，请使用以下检索到的上下文来回答问题，如果你不知道答案，就说你不知道。问题是：{question},上下文: {context},答案是:
"""
prompt = ChatPromptTemplate.from_template(template)

# Generate answer
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
#from langchain.chat_models import ChatOllama
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#llm = ChatOllama(model_name="mistral", temperature=0)
print("##flag1")
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("##flag2")
query = "萧炎的表妹是谁？"
res=rag_chain.invoke(query)
print("##flag3")
print(f'答案：{res}')
