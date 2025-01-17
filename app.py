import os
import time # to mesure execution time
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction( 
    api_key=openai_key, model_name="text-embedding-3-small" 
    )

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)

client = OpenAI(api_key=openai_key)

# Openai client test
# start_time = time.time()

# resp = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "What is human life expectancy in the Bangladesh?",
#         },
#     ],
# )

# print(resp)

# end_time = time.time()

# execution_time = end_time - start_time
# print(f"Execution time: {execution_time} seconds")

