import os
import json
import base64
import asyncio
import websockets
import logging
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

load_dotenv()

# Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_ENDPOINT = os.getenv("AZURE_OPENAI_API_ENDPOINT")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
AZURE_SEARCH_SEMANTIC_CONFIGURATION = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIGURATION")
VOICE = "alloy"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Azure Search Client Setup
credential = AzureKeyCredential(AZURE_SEARCH_KEY)
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX, credential=credential
)

# Streamlit App
def main():
    st.title("ATEA  AI-Assistant")
    handle_interface()

def handle_interface():
    """Handle the Streamlit interface."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello, I am Rose from Atea. How can I help you today?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Audio recorder
    audio_file = audio_recorder()

    if audio_file:
        asyncio.run(handle_audio_input(audio_file))

async def handle_audio_input(audio_file):
    """Handle audio input from Streamlit and send it to OpenAI."""
    async with websockets.connect(
        AZURE_OPENAI_API_ENDPOINT,
        additional_headers={"api-key": AZURE_OPENAI_API_KEY},
    ) as openai_ws:
        await initialize_session(openai_ws)

        # Send audio data to OpenAI
        audio_append = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_file).decode('utf-8')
        }
        await openai_ws.send(json.dumps(audio_append))

        # Receive response from OpenAI
        async for openai_message in openai_ws:
            response = json.loads(openai_message)

            if response.get("type") == "response.audio.delta" and "delta" in response:
                audio_payload = base64.b64decode(response["delta"])
                st.audio(audio_payload, format='audio/wav')

            # Handle function calls for RAG
            if response.get("type") == "response.function_call_arguments.done":
                function_name = response["name"]
                if function_name == "get_additional_context":
                    query = json.loads(response["arguments"]).get("query", "")
                    search_results = azure_search_rag(query)
                    logger.info(f"RAG Results: {search_results}")
                    await send_function_output(openai_ws, response["call_id"], search_results)

            # Trigger RAG search when input is committed
            if response.get("type") == "input_audio_buffer.committed":
                query = response.get("text", "").strip()
                if query:
                    logger.info(f"Triggering RAG search for query: {query}")
                    await trigger_rag_search(openai_ws, query)
                else:
                    logger.warning("Received empty query; skipping RAG search.")

async def initialize_session(openai_ws):
    """Initialize the OpenAI session with instructions and tools."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": (
                "You are an AI assistant providing factual answers ONLY from the search. "
                "If USER says hello Always respond with with Hello, I am Rose from Insurance Company. How can I help you today? "
                "Use the `get_additional_context` function to retrieve relevant information."
                "Keep all your responses very concise and straight to point and not more than 15 words"
                "If USER says Thank You,  Always respond with with You are welcome, Is there anything else I can help you with?"
            ),
            "tools": [
                {
                    "type": "function",
                    "name": "get_additional_context",
                    "description": "Fetch context from Azure Search based on a user query.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                }
            ],
        },
    }
    await openai_ws.send(json.dumps(session_update))

async def trigger_rag_search(openai_ws, query):
    """Trigger RAG search for a specific query."""
    search_function_call = {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call",
            "name": "get_additional_context",
            "arguments": {"query": query},
        },
    }
    await openai_ws.send(json.dumps(search_function_call))

async def send_function_output(openai_ws, call_id, output):
    """Send RAG results back to OpenAI."""
    response = {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        },
    }
    await openai_ws.send(json.dumps(response))

    # Prompt OpenAI to continue processing
    await openai_ws.send(json.dumps({"type": "response.create"}))

def azure_search_rag(query):
    """Perform Azure Cognitive Search and return results."""
    try:
        logger.info(f"Querying Azure Search with: {query}")
        results = search_client.search(
            search_text=query,
            top=2,
            query_type="semantic",
            semantic_configuration_name=AZURE_SEARCH_SEMANTIC_CONFIGURATION,
        )
        summarized_results = [doc.get("chunk", "No content available") for doc in results]
        if not summarized_results:
            return "No relevant information found in Azure Search."
        return "\n".join(summarized_results)
    except Exception as e:
        logger.error(f"Error in Azure Search: {e}")
        return "Error retrieving data from Azure Search."

if __name__ == "__main__":
    main()