from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import create_retriever_tool

load_dotenv()

# 1. Load a pretrained Sentence Transformer model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


texts=[
    'I think pears taste very good.',
    'I hate bananas.',
    'I love apples.',
    'I dislike raspberries.',
    'I enjoy oranges.',
    'I despise mangos.',
    'I love Linux.',
    'I hate Windows.'
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

print(vector_store.similarity_search('What fruits does a person like?',k=3))
print(vector_store.similarity_search('What fruits does a person hate?',k=3))

retriever = vector_store.as_retriever(search_kwargs={'k':3})

retriever_tool = create_retriever_tool(retriever, name='kb_search', description="search the small products / fruit database for information")

agent = create_agent(
    model = "groq:llama-3.3-70b-versatile",
    tools = [retriever_tool],
    system_prompt=("you are a helpful assistant. For questions about Macs, apples or laptops,"
    "first call the kb_search tool to retrive context, then answer succinctly. Maybe you have to use it multiple times before answering"
    )
)
response = agent.invoke({
    'messages':[
        {'role':'user','content':'What three fruits does a person like and What three fruits does a person dislike?'}
    ]
})

print(response)
print(response['messages'][-1].content)