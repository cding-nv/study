from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_community.vectorstores import DocArrayInMemorySearch

bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

vectordb = DocArrayInMemorySearch.from_texts(
    ["青蛙是食草动物",
     "人是由恐龙进化而来的。",
     "熊猫喜欢吃天鹅肉。",
     "1+1=5",
     "2+2=8",
     "3+3=9",
    "Gemini Pro is a Large Language Model was made by GoogleDeepMind",
     "A Language model is trained by predicting the next token"
    ],
    embedding=bge_embeddings
)

# #创建检索器
bge_retriever = vectordb.as_retriever(search_kwargs={"k": 1})

print(bge_retriever.get_relevant_documents("熊猫"))
print(bge_retriever.get_relevant_documents("恐龙"))
print(bge_retriever.get_relevant_documents("动物"))
print(bge_retriever.get_relevant_documents("Gemini"))
print(bge_retriever.get_relevant_documents("2+2"))
print(bge_retriever.get_relevant_documents("老虎"))
print(bge_retriever.get_relevant_documents("4+4"))
print(bge_retriever.get_relevant_documents("中国的首都是"))

