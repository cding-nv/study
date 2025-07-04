# Setup

```
conda create -n _baai python=3.10
conda activate _baai
pip install langchain_experimental langchain_core
pip install sentence_transformers
pip install -U langchain-community
pip install "langchain[docarray]"
pip install pydantic==1.10.9

python test_baai.py
```

Reference output:
```
[Document(page_content='熊猫喜欢吃天鹅肉。')]
[Document(page_content='人是由恐龙进化而来的。')]
[Document(page_content='青蛙是食草动物')]
[Document(page_content='Gemini Pro is a Large Language Model was made by GoogleDeepMind')]
[Document(page_content='2+2=8')]
[Document(page_content='3+3=9')]
[Document(page_content='Gemini Pro is a Large Language Model was made by GoogleDeepMind')]
[Document(page_content='1+1=5')]
```

# Model download
   https://github.com/FlagOpen/FlagEmbedding
