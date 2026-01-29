from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()

# 1. Load a pretrained Sentence Transformer model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


texts=[
    'Apple makes very good computers.',
    'I believe Apple is innovative!',
    'I love apples.',
    'I am a big fan of Macbooks.',
    'I enjoy oranges.',
    'I like levono Thinkpads.',
    'I think pears taste very good.'
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

# print(vector_store.similarity_search('Apples are my favourite food.',k=7))
print(vector_store.similarity_search('Linux is a great Operating system.',k=7))
