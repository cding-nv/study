from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
from intel_extension_for_transformers.neural_chat import plugins
from intel_extension_for_transformers.transformers import MixedPrecisionConfig

plugins.retrieval.enable=True

#include if security checker needed
#plugins.safety_checker.enable = True

# Path to cleaned txt file
#plugins.retrieval.args["input_path"]="./txt_files-zh/"
plugins.retrieval.args["input_path"]="./txt_files-en/"
plugins.retrieval.args["vector_database"]="Chroma"

# Choose embedding model
#plugins.retrieval.args["embedding_model"]="sentence-transformers/all-mpnet-base-v2"
plugins.retrieval.args["embedding_model"]="BAAI/bge-base-en-v1.5"
#plugins.retrieval.args["embedding_model"]="BAAI/bge-base-zh-v1.5"

# replace the template with the answer you want when related document not found
plugins.retrieval.args["response_template"]="Can you please rephrase the question, I am not able to fetch the response for your question."

plugins.retrieval.args["mode"]="accuracy"
plugins.retrieval.args["search_type"]="similarity"
plugins.retrieval.args["append"]=False

#can be looked at later if needed
#plugins.retrieval.args["retrieval_type"]="child_parent"
#plugins.retrieval.args["max_chuck_size"]=1000
#plugins.retrieval.args["process"]=True

config = PipelineConfig(plugins=plugins, optimization_config=MixedPrecisionConfig())
#config = PipelineConfig(model_name_or_path="Qwen/Qwen-7B", plugins=plugins, optimization_config=MixedPrecisionConfig())
chatbot = build_chatbot(config)

#response = chatbot.predict(query="英特尔第八任首席执行官是谁?")
#print(response)

response = chatbot.predict(query="What is FasterTransformer?")
print(response)

response = chatbot.predict(query="What is FasterTransformer built on?")
print(response)

response = chatbot.predict(query="What is Intel extension for pytorch?")
print(response)
