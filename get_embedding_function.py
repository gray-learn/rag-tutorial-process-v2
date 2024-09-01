from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings

def get_embedding_function():
    embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
    return embeddings

