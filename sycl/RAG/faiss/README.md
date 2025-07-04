# Faiss with Langchain on Intel GPU

[Faiss](https://github.com/facebookresearch/faiss) library is migrated to SYCL with Intel DPC++ Compatibility Tools (DPCT). And generate python wheel here [faiss-1.8.0-py3-none-any.whl](./faiss-1.8.0-py3-none-any.whl)

Experiment platforms: `dGPU Max 1100`, `ARC770`, `Raptor Lake iGPU`    
oneAPI: 2024.2, 2025.0

## How to Run

### 1. Setup Env

For example, start a docker container with intel GPU:
```
sudo docker run --rm -it --privileged --device=/dev/dri --ipc=host --net=host -v /home/fengding:/home/fengding intel/intel-extension-for-pytorch:2.1.30-xpu bash
```

### 2. Requirements.

```
pip install -qU langchain-community
pip install sentence_transformers
pip install pydantic==1.10.9
pip install docarray

pip install ./faiss-1.8.0-py3-none-any.whl
```

### 3. Run
Download [state_of_the_union.txt](https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/document_loaders/example_data/state_of_the_union.txt) which is used in test_langchain.py

```
$ source intel/oneapi/setvars.sh
$ export SYCL_CACHE_PERSISTENT=1
$ python test_langchain.py
```
Reference output:
```
........
Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections.

Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.

One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.

And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.

```

### 4. Code reference
[langchain_community/vectorstores/faiss.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py)
```
from langchain_community.vectorstores import FAISS
        --> langchain_community/vectorstores/faiss.py 
            --> dependable_faiss_import() --> import faiss
```

## Other python examples also can work
https://github.com/facebookresearch/faiss/tree/main/tutorial/python
