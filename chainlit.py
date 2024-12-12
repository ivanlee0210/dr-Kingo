import os
import openai
import chainlit as cl
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager
from llama_index.vector_stores.supabase import SupabaseVectorStore

from loguru import logger

load_dotenv()

aoi_api_key = os.getenv("AZURE_OPENAI_API_KEY")
aoi_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
aoi_gen_model = os.getenv("AZURE_GENERATION_MODEL")
aoi_version = os.getenv("AZURE_GENERATION_MODEL_VERSION")
aoi_emb_model = os.getenv("AZURE_EMBEDDING_MODEL")

openai.api_key = os.environ.get("OPENAI_API_KEY")
supabase_url = os.getenv("DATABASE_URL")

## Put today's date in a string 
now = datetime.now()
current_date_str = now.strftime("%A %Y-%m-%d")

# # Old date format
# current_date_str = now.strftime("%A %B %d, %Y")

llm = AzureOpenAI(
    model=aoi_gen_model,
    deployment_name=aoi_gen_model,
    azure_endpoint=aoi_endpoint,
    api_key=aoi_api_key,
    api_version=aoi_version,
    temperature=0.2,
        system_prompt=f"""You are a chatbot on Sungkyunkwan University website. The current date and time in Korea is {current_date_str}. Provide detailed answers in a structured Markdown format using all relevant information available in the context.

    Rules:
    1. Today is {current_date_str}
    2. Use markdown headings, lists, and tables to structure your output.
    3. When providing menu information, always format it in a Markdown table with the following columns:
        - Dish
        - Price
        - Restaurant Name

        For example:

        | Dish                 | Price   | Restaurant Name |
        |----------------------|---------|-----------------|
        | Soft Tofu Stew       | 8,000KRW| Gusijae         |
        | Pork and Kimchi Stew | 7,000KRW| Eunhaenggol     |
    4. Avoid saying "based on the context." Provide answers directly as if you retrieved the information yourself.
    5. If the user asks about the shuttle bus, mention where the bus stops are. Plus, one can see the live location of the campus shuttle bus between Hyehwa Station and the Seoul campus on this website: http://route.hellobus.co.kr:8787/pub/routeView/skku/bis_view.html. Include this link in your response if relevant.
    6. Try to answer in the same style and language as the user's query.
    7. Only include markdown formatting and do no include HTML tags in your response.
    8. When answering questions about restaurants and food menus, conclude the response with a statement that the user can find more information on the official SKKU website: Dining Halls/Menu for [Humanities and Social Science](https://www.skku.edu/eng/CampusLife/convenience/DiningHallsMenu.do) and [Natural Sciences](https://www.skku.edu/eng/CampusLife/convenience/DiningHallsMenu02.do)
    9. When answering questions about upcoming events, always mention put the nearest event first.
    10. Make sure that urls start with https://www.skku.edu/ if you include them in your response.
    """
)

embed_model = AzureOpenAIEmbedding(
    model=aoi_emb_model,
    deployment_name=aoi_emb_model,
    api_key=aoi_api_key,
    azure_endpoint=aoi_endpoint,
    api_version=aoi_version,
)


Settings.llm = llm
Settings.embed_model = embed_model
Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])

# Create the vector store and storage context with detailed logging
logger.debug("Creating SupabaseVectorStore with connection string: %s and collection name: %s", supabase_url, "md_kingo")
vector_store = SupabaseVectorStore(
    postgres_connection_string=supabase_url, collection_name="md_kingo",
)
logger.debug("SupabaseVectorStore created: %s", vector_store)

logger.debug("Creating StorageContext with the vector store")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
logger.debug("StorageContext created: %s", storage_context)

# load your index from stored vectors
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="When winter break?",
            message=f"When does the semester ends, and the winter break starts?",
            icon="https://img.icons8.com/?size=100&id=28WzaxCdZhVK&format=png&color=000000",
            ),

        cl.Starter(
            label="Where is the campus shuttle bus?",
            message="Tell me where the Seoul campus shuttle bus is right now",
            icon="https://img.icons8.com/?size=100&id=QyVp2C1vt266&format=png&color=000000",
            ),
        cl.Starter(
            label="셔틀버스 정보",
            message=f"지금 셔틀버스 어디에 있나요",
            icon="https://img.icons8.com/?size=100&id=QyVp2C1vt266&format=png&color=000000",
            ),
        cl.Starter(
            label="When submit thesis?",
            message=f"I am graduating in February, when do I need to submit my thesis?",
            icon="https://img.icons8.com/?size=100&id=28WzaxCdZhVK&format=png&color=000000",
            ),
        ]

@cl.on_chat_start
async def start():
    query_engine = index.as_chat_engine(streaming=True, similarity_top_k=20, chat_mode="condense_question", use_async=True)
    cl.user_session.set("query_engine", query_engine)



@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine") # type: RetrieverQueryEngine

    response_message = cl.Message(content="", author="dr Kingo")

    response = await cl.make_async(query_engine.chat)(message.content)

    for token in response.response:
        logger.debug("Streaming token: %s", token)
        await response_message.stream_token(token=token)
    
    await response_message.send()
    logger.debug("Sent response message")
