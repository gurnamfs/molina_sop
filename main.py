import io
import re
from contextlib import redirect_stdout
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# from functions import initial_checks, final
from functions import initial_checks
from summarize import summary
# from functions import initial_checks
from tool_func import langgraph_agent_executor, guide_tool
from prompts import sys
# from tool_func import messages
import logging
from fastapi.responses import StreamingResponse
import asyncio
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,  # Change this to DEBUG for more detailed #logging
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# async def data_streamer(query,file_path):
#     guide_tool.clear()
#     logging.info(f"Received Request for SOP Navigation with query: {query} and file path: {file_path}")
#     final_message = ""
#     processed_claim = initial_checks(query)
#     guide_tool.append(processed_claim)
#     inp = f"Please process create_agent(file_path= {file_path}, query= {processed_claim})"
    
#     async for msg, metadata in langgraph_agent_executor.astream(
#         {"messages": [("user", inp)]}, stream_mode="messages"
#     ):
#         if (
#             msg.content
#             and not isinstance(msg, HumanMessage)
#             and metadata["langgraph_node"] == "tools"
#             and not msg.name
#         ):
#             yield str(msg.content)  
        
#         if msg.content and metadata["langgraph_node"] == "agent":
#             final_message += msg.content
        
#     output_stream = io.StringIO()
#     with redirect_stdout(output_stream):
#         res = final(processed_claim)
#     verbose_output = output_stream.getvalue()
#     logging.info(f"Timely filing Processed")
#     verbose_output = verbose_output.replace("Entering new AgentExecutor chain...", "").replace("Action: json_spec_list_keys", "").replace("Action: json_spec_get_value", "").replace("ValueError", "")
        
#     cleaned_text = re.sub(
#             r"\u001b\[\d+;\d+m|\u001b\[0m|\u001b\[1m>", "", verbose_output
#         )
    
#     yield final_message + "\n" + cleaned_text


async def data_streamer(query,file_path):
    guide_tool.clear()
    logging.info(f"Received Request for SOP Navigation with query: {query} and file path: {file_path}")
    final_message = ""
    processed_claim = initial_checks(query)
    guide_tool.append(processed_claim)
    inp = f"Please process create_agent(file_path= {file_path}, query= {processed_claim})"
    
    async for msg, metadata in langgraph_agent_executor.astream(
        {"messages": [("user", inp)]}, stream_mode="messages"
    ):
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and metadata["langgraph_node"] == "tools"
            and not msg.name
        ):
            yield str(msg.content)  
        
        if msg.content and metadata["langgraph_node"] == "agent":
            final_message += msg.content

    yield final_message



@app.post("/sop_navigation_stream/")
async def sop_navigation(
    query, file_path="Coordination of Benefits (COB) - All States Medicaid - SOP.json"
):
    
    try:
        return StreamingResponse(data_streamer(query,file_path), media_type='text/event-stream')

    except Exception as e:
        logging.error(f"Error during SOP execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/sop_navigation/")
async def sop_navigation(
    query, file_path="Coordination of Benefits (COB) - All States Medicaid - SOP.json"
):
    logging.info(f"Received Request for SOP Navigation with query: {query} and file path: {file_path}")

    try:
        # messages.clear()
        # messages.append(("system", sys))
        # logging.debug("Messages cleared and system message added.")
        guide_tool.clear()
        processed_claim = initial_checks(query)
        guide_tool.append(processed_claim)
        logging.info(f"Processed claim")
        # inp = f"file_path: json-files/{file_path}, Claim: {processed_claim}"
        # inp = f"Please Process create_agent(file_path = json-files/{file_path}, query = {query})"
        inp = f"Please process create_agent(file_path = json-files/{file_path}, query = {processed_claim})"
        logging.debug(f"First Input to LangGraph agent: {inp}")

        output_stream = io.StringIO()
        with redirect_stdout(output_stream):
            res = langgraph_agent_executor.invoke({"messages": [("user", inp)]})
        verbose_output = output_stream.getvalue()
        logging.debug(f"Verbose output fetched")
        verbose_output = (
            verbose_output.replace("Action: json_spec_list_keys", "")
            .replace("Action: json_spec_get_value", "")
            .replace("ValueError", "")
        )
        cleaned_text = re.sub(
            r"\u001b\[\d+;\d+m|\u001b\[0m|\u001b\[1m>", "", verbose_output
        )
        logging.debug("Verbose output cleaned.")
        return cleaned_text 

    except Exception as e:
        logging.error(f"Error during SOP execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/summarize_navigation/")
# async def summarize_navigation(steps):
#     result = summary(steps)
#     return {"Summary": result}


@app.post("/summarize_navigation/")
async def summarize_navigation(steps):
    return await summary(steps)