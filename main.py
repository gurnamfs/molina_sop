import io
import re
from contextlib import redirect_stdout
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from functions import initial_checks, final
from tool_func import langgraph_agent_executor
from prompts import sys
from tool_func import messages
import logging
from fastapi.responses import StreamingResponse
import asyncio
from langchain_core.messages import AIMessageChunk, HumanMessage

logging.basicConfig(
    level=logging.INFO,  # Change this to DEBUG for more detailed logging
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


async def data_streamer(query,file_path):
    logging.info(f"Received Request for SOP Navigation with query: {query} and file path: {file_path}")
    final_message = ""
    messages.clear()
    messages.append(("system", sys))
    logging.debug("Messages cleared and system message added.")
    processed_claim = initial_checks(query)
    res = final(processed_claim)["output"]
    logging.info(f"Timely filing Processed")
    logging.info(f"Processed claim")
    inp = f"file_path: json-files/{file_path}, Claim: {processed_claim}"
    logging.debug(f"First Input to LangGraph agent: {inp}")
    
    async for msg, metadata in langgraph_agent_executor.astream(
        {"messages": [("user", inp)]}, stream_mode="messages"
    ):
        # Stream all messages from the tool node
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and metadata["langgraph_node"] == "tools"
            and not msg.name
        ):
            yield str(msg.content)  # Yielding the tool message content
            # await asyncio.sleep(0.5)  # Simulate a delay
        
        # Final message should come from our agent
        if msg.content and metadata["langgraph_node"] == "agent":
            final_message += msg.content
    
    # Yield the final message from the agent
    yield final_message + "\n" + res



@app.post("/sop_navigation_stream/")
async def sop_navigation_stream(
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
        messages.clear()
        messages.append(("system", sys))
        logging.debug("Messages cleared and system message added.")
        processed_claim = initial_checks(query)
        logging.info(f"Processed claim")
        inp = f"file_path: json-files/{file_path}, Claim: {processed_claim}"
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
        res = final(processed_claim)["output"]
        logging.info(f"Timely filing Processed")
        return cleaned_text + f"\n\nFinal Answer: {res}"

    except Exception as e:
        logging.error(f"Error during SOP execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
