import streamlit as st
import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import requests
import time
import json
import pandas as pd

load_dotenv()

st.set_page_config(page_title="Chat & Upload App", page_icon="💬", layout="wide")

st.title("💬 Chat and Document Upload")
st.markdown("A simple Streamlit interface featuring a chat window and file uploader.")


# Initialize clients and models
@st.cache_resource
def init_services():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    from pinecone import ServerlessSpec
    
    index_name = "smfinance-index"
    if index_name not in pc.list_indexes().names():
        print(f"Index {index_name} not found. Creating it now...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
            
    index = pc.Index(index_name)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return index, model, openai_client

index, model, openai_client = init_services()

def upload_to_azure(file_bytes, file_name):
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_CONTAINER_NAME")
    if not connection_string or not container_name:
        st.error("Azure Storage credentials are not set in .env")
        return False
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
        blob_client.upload_blob(file_bytes, overwrite=True)
        return True
    except Exception as e:
        st.error(f"Error uploading to Azure: {e}")
        return False

# Initialize chat history and state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# Sidebar for file upload
with st.sidebar:
    st.header("📂 Upload Documents")
    
    # Check if there is an active document from query params across refresh
    if "file" in st.query_params and "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = st.query_params["file"]

    # Display active document
    if "uploaded_file_name" in st.session_state:
        st.success(f"Active Document: {st.session_state.uploaded_file_name}")
        if st.button("Remove Active Document"):
            del st.session_state["uploaded_file_name"]
            if "file" in st.query_params:
                del st.query_params["file"]
            st.session_state["uploader_key"] += 1
            st.rerun()

    uploaded_file = st.file_uploader(
        "Upload a new file for context" if "uploaded_file_name" in st.session_state else "Upload a file for context", 
        type=["txt", "csv", "pdf", "docx", "png", "jpg", "jpeg"],
        key=f"uploader_{st.session_state['uploader_key']}"
    )
    
    if uploaded_file is not None:
        if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            with st.spinner("Uploading to Azure Storage..."):
                file_bytes = uploaded_file.getvalue()
                success = upload_to_azure(file_bytes, uploaded_file.name)
                
            if success:
                st.success(f"Successfully uploaded to Azure: {uploaded_file.name}")
                
                # Trigger DataBricks Job and Wait
                with st.status("Triggering Fivetran and Databricks processing pipeline...", expanded=True) as status:
                    st.write("Initiating processing job...")
                    
                    dbx_host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
                    dbx_token = os.getenv("DATABRICKS_TOKEN")
                    job_id = os.getenv("DATABRICKS_JOB_ID")
                    
                    if dbx_host and dbx_token and job_id:
                        headers = {"Authorization": f"Bearer {dbx_token}"}
                        trigger_url = f"{dbx_host}/api/2.1/jobs/run-now"
                        payload = {
                            "job_id": int(job_id),
                            "notebook_params": {"blob_name": uploaded_file.name}
                        }
                        
                        try:
                            res = requests.post(trigger_url, headers=headers, json=payload)
                            res.raise_for_status()
                            run_id = res.json().get("run_id")
                            st.write(f"Job triggered successfully (Run ID: {run_id}). Waiting for Fivetran sync and Databricks vectorization...")
                            
                            # Polling loop
                            status_url = f"{dbx_host}/api/2.1/jobs/runs/get"
                            while True:
                                stat_res = requests.get(status_url, headers=headers, params={"run_id": run_id})
                                stat_res.raise_for_status()
                                run_state = stat_res.json().get("state", {})
                                life_cycle_state = run_state.get("life_cycle_state")
                                result_state = run_state.get("result_state")
                                
                                if life_cycle_state in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
                                    if result_state == "SUCCESS":
                                        status.update(label="Processing complete! Ready to chat.", state="complete", expanded=False)
                                        st.session_state.uploaded_file_name = uploaded_file.name
                                        st.query_params["file"] = uploaded_file.name
                                        break
                                    else:
                                        status.update(label=f"Job failed with status: {result_state}", state="error", expanded=True)
                                        st.error(run_state.get("state_message", "Unknown Databricks error"))
                                        break
                                time.sleep(5)
                                
                        except Exception as e:
                            status.update(label="Failed to trigger processing job", state="error", expanded=True)
                            st.error(f"Failed calling Databricks API: {e}")
                    else:
                        status.update(label="Missing Databricks ENV credentials. Skipping API trigger.", state="error")
                        st.error("Please set DATABRICKS_HOST, DATABRICKS_TOKEN, and DATABRICKS_JOB_ID in .env.")
                        st.session_state.uploaded_file_name = uploaded_file.name
                        st.query_params["file"] = uploaded_file.name

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("content"):
            st.markdown(message["content"])
        if message.get("chart"):
            chart_info = message["chart"]
            st.markdown(f"**{chart_info.get('title', 'Analytics')}**")
            df = pd.DataFrame(chart_info['data'])
            chart_type = chart_info.get("type", "bar")
            if chart_type == "bar":
                st.bar_chart(df, x="x", y="y")
            elif chart_type == "line":
                st.line_chart(df, x="x", y="y")
            elif chart_type == "scatter":
                st.scatter_chart(df, x="x", y="y")
            else:
                st.bar_chart(df, x="x", y="y")

# React to user input
if prompt := st.chat_input("Type your message here..."):
    if "uploaded_file_name" not in st.session_state:
        st.warning("Please upload a document from the sidebar to begin chatting.")
        st.stop()
        
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve from Pinecone
    with st.spinner("Thinking..."):
        try:
            # Step 1: embed the question
            question_embedding = model.encode(prompt).tolist()
            
            # Step 2: search Pinecone
            results = index.query(
                vector=question_embedding,
                top_k=1000,
                include_metadata=True,
                filter={"source": st.session_state.uploaded_file_name}
            )
            
            # Step 3: extract context
            context_texts = [r["metadata"]["text"] for r in results["matches"] if "text" in r["metadata"]]
            context = "\n\n".join(context_texts)
            
            # Step 4: Call LLM
            system_prompt = (
                "You are an AI data assistant. Use the following context retrieved from the user's document "
                "database to answer the user's question. If the answer is not in the context, say you don't know.\n\n"
                f"Context:\n{context}"
            )

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "render_chart",
                        "description": "Renders a data visualization chart. Use this when the user asks for analytics, charts, or graphs based on numerical data in the context.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "chart_type": {
                                    "type": "string",
                                    "enum": ["bar", "line", "scatter"],
                                    "description": "The type of chart to display."
                                },
                                "title": {
                                    "type": "string",
                                    "description": "The title of the chart."
                                },
                                "data": {
                                    "type": "array",
                                    "description": "The data points to plot.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "x": {
                                                "type": "string",
                                                "description": "The label or category for the X-axis (e.g., date, name)."
                                            },
                                            "y": {
                                                "type": "number",
                                                "description": "The numerical value for the Y-axis."
                                            }
                                        },
                                        "required": ["x", "y"]
                                    }
                                }
                            },
                            "required": ["chart_type", "title", "data"]
                        }
                    }
                }
            ]

            messages = [{"role": "system", "content": system_prompt}]
            for msg in st.session_state.messages:
                if msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            chat_completion = openai_client.chat.completions.create(
                model="gpt-4o", # Better model for reasoning & function calling
                messages=messages,
                stream=True,
                tools=tools
            )
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                tool_calls = []
                
                # Stream the response
                for chunk in chat_completion:
                    delta = chunk.choices[0].delta
                    if delta.content is not None:
                        full_response += delta.content
                        message_placeholder.markdown(full_response + "▌")
                        
                    if delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if len(tool_calls) <= tool_call.index:
                                tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                            
                            if tool_call.id:
                                tool_calls[tool_call.index]["id"] = tool_call.id
                            if tool_call.type:
                                tool_calls[tool_call.index]["type"] = tool_call.type
                            if tool_call.function:
                                if tool_call.function.name:
                                    tool_calls[tool_call.index]["function"]["name"] = tool_call.function.name
                                if tool_call.function.arguments:
                                    tool_calls[tool_call.index]["function"]["arguments"] += tool_call.function.arguments

                if full_response:
                    message_placeholder.markdown(full_response)
                else:
                    message_placeholder.empty()

                chart_data_to_store = None
                if tool_calls:
                    for tc in tool_calls:
                        if tc["function"]["name"] == "render_chart":
                            try:
                                args = json.loads(tc["function"]["arguments"])
                                chart_type = args.get("chart_type", "bar")
                                data = args.get("data", [])
                                title = args.get("title", "Analytics")
                                
                                if data:
                                    df = pd.DataFrame(data)
                                    st.markdown(f"**{title}**")
                                    if chart_type == "bar":
                                        st.bar_chart(df, x="x", y="y")
                                    elif chart_type == "line":
                                        st.line_chart(df, x="x", y="y")
                                    elif chart_type == "scatter":
                                        st.scatter_chart(df, x="x", y="y")
                                    else:
                                        st.bar_chart(df, x="x", y="y")
                                        
                                    chart_data_to_store = {
                                        "title": title,
                                        "type": chart_type,
                                        "data": data
                                    }
                            except Exception as e:
                                st.error(f"Failed to parse or render chart data: {e}")
                
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "chart": chart_data_to_store
            })
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
