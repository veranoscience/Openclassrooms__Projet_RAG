import sys

print ("test environement")

print ("test import faiss")
import faiss
print ("faiss importé avec succès")

print ("test import faiss langchain")
from langchain_community.vectorstores import FAISS
print ("faiss langchaoin importé avec succès")

print ("test import huggingface embdings")
from langchain_huggingface import HuggingFaceEmbeddings
print ("hugging face embeddings importé avec succès")

print ("test import mistral")
from mistralai import Mistral
print ("mistralai importé avec succès")