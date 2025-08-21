# Travel chatbot with LangGraph and Gradio UI, adapted from 4_lab4.ipynb.
# Key features:
# - Uses LangGraph (like Cell 17) to define a graph with nodes (extractor, validator) and conditional edges.
# - Extracts flight details using structured outputs (like EvaluatorOutput in Cell 7).
# - Validates details, with iterative refinement, mimicking notebook's worker-evaluator loop.
# - Gradio UI for interactivity, like Cell 22.
# - Why LangGraph? Provides modularity, state management, and scalability; aligns with notebook's multi-agent design.
# - Requirements: langchain-google-genai, pydantic, python-dotenv, gradio, langgraph.
# - Set GOOGLE_API_KEY in .env file.

from typing import TypedDict, Optional, Annotated, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import gradio as gr
import uuid
import asyncio

# Load environment variables (like Cell 3)
# Why? Securely handles API keys, avoids hardcoding.
load_dotenv(override=True)
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Initialize LLMs (like Cell 5)
# Why two LLMs? Separates extraction and validation for modularity, like worker/evaluator.
extractor_llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
validator_llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)

# Define Pydantic schemas (like EvaluatorOutput in Cell 7)
# Why Pydantic? Enforces structured LLM outputs, ensuring parseable results.
class FlightQueryDetails(BaseModel):
    departure_city: Optional[str] = Field(description="Departure city/airport (e.g., 'New York', 'JFK')")
    arrival_city: Optional[str] = Field(description="Arrival city/airport (e.g., 'Los Angeles', 'LAX')")
    departure_date: Optional[str] = Field(description="Departure date in YYYY-MM-DD; infer if not explicit")
    return_date: Optional[str] = Field(description="Return date in YYYY-MM-DD; None for one-way")
    num_passengers: Optional[int] = Field(description="Number of passengers; default 1")
    travel_class: Optional[str] = Field(description="Class: 'economy', 'business', 'first'; None if unspecified")
    additional_notes: Optional[str] = Field(description="Other preferences or details")

class ValidationResult(BaseModel):
    is_complete: bool = Field(description="True if required fields (departure_city, arrival_city, departure_date) are present")
    feedback: str = Field(description="Feedback on missing/invalid fields or clarification needed")
    needs_refinement: bool = Field(description="True if query needs reprocessing")

# Bind LLMs to structured outputs (like Cell 11)
# Why? Ensures LLMs return Pydantic models, like evaluator_llm_with_output.
extractor_llm_with_output = extractor_llm.with_structured_output(FlightQueryDetails)
validator_llm_with_output = validator_llm.with_structured_output(ValidationResult)

# Custom reducer for messages (like add_messages in Cell 9)
# Why? Appends messages to maintain history, critical for context in refinement.
def add_messages_to_state(left: List[AIMessage | HumanMessage], right: List[AIMessage | HumanMessage]) -> List[AIMessage | HumanMessage]:
    return left + right

# Define state (like Cell 9)
# Why? Tracks query, messages, extracted details, validation, and iterations for stateful processing.
class ExtractionState(TypedDict):
    user_query: str
    messages: Annotated[List[AIMessage | HumanMessage], add_messages_to_state]
    extracted_details: Optional[FlightQueryDetails]
    validation_result: Optional[ValidationResult]
    iteration_count: int

# Helper: Infer date for vague terms (e.g., "next Friday")
# Why? Handles natural language queries, adding robustness like notebook's worker clarifications.
def infer_date(relative_date: str) -> Optional[str]:
    today = datetime.now()
    if "next friday" in relative_date.lower():
        days_until_friday = (4 - today.weekday() + 7) % 7 or 7  # 4 is Friday
        return (today + timedelta(days=days_until_friday)).strftime("%Y-%m-%d")
    return None

# Extractor node (like worker in Cell 12)
# Why? Processes query with LLM, returns structured details, updates state.
def extract_flight_details(state: ExtractionState) -> ExtractionState:
    system_message = """
    You are an expert flight query parser. Extract details from the user's query:
    - Departure/arrival cities or airports
    - Departure/return dates (YYYY-MM-DD; infer if vague, e.g., 'next Friday')
    - Number of passengers (default 1)
    - Travel class (economy, business, first)
    - Additional notes (e.g., preferences)
    Set missing fields to None. Always return structured output.
    """
    messages = state["messages"] + [SystemMessage(content=system_message), HumanMessage(content=state["user_query"])]
    try:
        details = extractor_llm_with_output.invoke(messages)
        if details.departure_date and "next" in details.departure_date.lower():
            details.departure_date = infer_date(details.departure_date)
        return {
            "user_query": state["user_query"],
            "messages": messages + [AIMessage(content=str(details))],
            "extracted_details": details,
            "validation_result": None,
            "iteration_count": state["iteration_count"]
        }
    except Exception as e:
        return {
            "user_query": state["user_query"],
            "messages": messages + [AIMessage(content=f"Extraction error: {str(e)}")],
            "extracted_details": None,
            "validation_result": None,
            "iteration_count": state["iteration_count"]
        }

# Validator node (like evaluator in Cell 15)
# Why? Checks extracted details, provides feedback, drives refinement.
def validate_flight_details(state: ExtractionState) -> ExtractionState:
    if not state["extracted_details"]:
        return {
            "user_query": state["user_query"],
            "messages": state["messages"] + [AIMessage(content="Validation failed: No extracted details")],
            "extracted_details": None,
            "validation_result": ValidationResult(
                is_complete=False,
                feedback="No details extracted; please clarify the query.",
                needs_refinement=True
            ),
            "iteration_count": state["iteration_count"] + 1
        }
    details = state["extracted_details"]
    missing_fields = []
    if not details.departure_city:
        missing_fields.append("departure city")
    if not details.arrival_city:
        missing_fields.append("arrival city")
    if not details.departure_date:
        missing_fields.append("departure date")
    system_message = """
    You are a validator for flight query details. Check if the extracted details are complete:
    - Required: departure_city, arrival_city, departure_date
    - Optional: return_date, num_passengers, travel_class, additional_notes
    Provide feedback and decide if refinement is needed.
    """
    user_message = f"""
    Extracted details:
    - Departure: {details.departure_city}
    - Arrival: {details.arrival_city}
    - Departure Date: {details.departure_date}
    - Return Date: {details.return_date}
    - Passengers: {details.num_passengers}
    - Class: {details.travel_class}
    - Notes: {details.additional_notes}
    Are these details complete? Provide feedback.
    """
    messages = [SystemMessage(content=system_message), HumanMessage(content=user_message)]
    try:
        validation = validator_llm_with_output.invoke(messages)
        if missing_fields:
            validation = ValidationResult(
                is_complete=False,
                feedback=f"Missing required fields: {', '.join(missing_fields)}. Please clarify.",
                needs_refinement=True
            )
        return {
            "user_query": state["user_query"],
            "messages": state["messages"] + [AIMessage(content=str(validation))],
            "extracted_details": state["extracted_details"],
            "validation_result": validation,
            "iteration_count": state["iteration_count"] + 1
        }
    except Exception as e:
        return {
            "user_query": state["user_query"],
            "messages": state["messages"] + [AIMessage(content=f"Validation error: {str(e)}")],
            "extracted_details": state["extracted_details"],
            "validation_result": ValidationResult(
                is_complete=False,
                feedback=f"Validation failed: {str(e)}",
                needs_refinement=True
            ),
            "iteration_count": state["iteration_count"] + 1
        }

# Router function (like Cells 13/16)
# Why? Defines conditional edges, controls flow like notebook's worker_router and route_based_on_evaluation.
def router(state: ExtractionState) -> str:
    if state["iteration_count"] >= 3:  # Prevent infinite loops
        return END
    if state["validation_result"] and state["validation_result"].is_complete:
        return END
    if state["validation_result"] and state["validation_result"].needs_refinement:
        return "extract"
    return "validate"

# Build and compile graph (like Cell 17)
# Why? Defines workflow as nodes and edges, ensures stateful processing with persistence.
workflow = StateGraph(ExtractionState)
workflow.add_node("extract", extract_flight_details)
workflow.add_node("validate", validate_flight_details)
workflow.add_edge("extract", "validate")  # Always validate after extraction
workflow.add_conditional_edges("validate", router, {"extract": "extract", END: END})
workflow.set_entry_point("extract")
checkpointer = MemorySaver()  # Like notebook, persists state per thread
graph = workflow.compile(checkpointer=checkpointer)

# Gradio callback (like Cell 21)
# Why? Handles async UI submissions, uses graph for processing, maintains thread for session isolation.
async def process_message(message: str, history: list, thread: str) -> tuple[list, str]:
    if not message:
        return history, thread
    state = await graph.ainvoke(
        {"user_query": message, "messages": [], "extracted_details": None, "validation_result": None, "iteration_count": 0},
        config={"configurable": {"thread_id": thread}}
    )
    # Format output for Gradio (like Cell 21)
    history.append({"role": "user", "content": message})
    if state["extracted_details"]:
        details = state["extracted_details"]
        details_str = (
            f"Extracted Details:\n"
            f"- Departure: {details.departure_city}\n"
            f"- Arrival: {details.arrival_city}\n"
            f"- Departure Date: {details.departure_date}\n"
            f"- Return Date: {details.return_date}\n"
            f"- Passengers: {details.num_passengers}\n"
            f"- Class: {details.travel_class}\n"
            f"- Notes: {details.additional_notes}"
        )
        history.append({"role": "assistant", "content": details_str})
    if state["validation_result"]:
        validation_str = (
            f"Validation Result:\n"
            f"- Complete: {state['validation_result'].is_complete}\n"
            f"- Feedback: {state['validation_result'].feedback}\n"
            f"- Needs Refinement: {state['validation_result'].needs_refinement}\n"
            f"- Iterations: {state['iteration_count']}"
        )
        history.append({"role": "assistant", "content": validation_str})
    return history, thread

# Reset function (like Cell 21)
# Why? Clears inputs and starts new session with a new thread ID.
async def reset() -> tuple[str, list, str]:
    return "", [], str(uuid.uuid4())

# Gradio UI (like Cell 22)
# Why? Interactive chatbot interface, user-friendly, mirrors Sidekick UI.
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("## Travel Chatbot")
    thread = gr.State(value=str(uuid.uuid4()))  # Session ID, like notebook
    with gr.Row():
        chatbot = gr.Chatbot(label="Travel Assistant", height=400, type="messages")
    with gr.Group():
        with gr.Row():
            message = gr.Textbox(show_label=False, placeholder="Enter your flight query (e.g., 'Fly from NYC to LA on 2023-12-15')")
    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        go_button = gr.Button("Go!", variant="primary")
    message.submit(process_message, [message, chatbot, thread], [chatbot, thread])
    go_button.click(process_message, [message, chatbot, thread], [chatbot, thread])
    reset_button.click(reset, [], [message, chatbot, thread])

# Launch Gradio app
# Why? Runs web server; local URL for testing, like notebook's demo.launch().
if __name__ == "__main__":
    demo.launch()