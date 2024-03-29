from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL = LangchainEmbedding(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
