from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import datetime
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from agents.tools.flights_finder import flights_finder
from agents.tools.hotels_finder import hotels_finder
from typing import Optional

_ = load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CURRENT_YEAR = datetime.datetime.now().year

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)
tools = [flights_finder, hotels_finder]
tool_names = ", ".join([t.name for t in tools])

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    (
        "You are a smart travel agency AI. Your job is to help users find flights and hotels.\n"
        "Follow this flow strictly:\n"
        "1) Track flight search requirements in the `flight_params` state, which includes: `departure_id`, `arrival_id`, `outbound_date`, `return_date` (optional for one-way), `type` ('1' for round trip, '2' for one-way, '3' for multi-city), `travel_class`, `adults`, `children`, `infants_in_seat`, `infants_on_lap`, `stops`, and optional preferences (`include_airlines`, `exclude_airlines`, `max_price`, `outbound_times`, `return_times`, `max_duration`, `multi_city_json`).\n"
        "2) From the conversation history and latest user input, extract any provided flight parameters and update `flight_params`. Required fields are: `departure_id`, `arrival_id`, `outbound_date`. For round trips, `return_date` is also required. Default `type` to '1' (round trip) unless specified.\n"
        "3) If any required fields are missing, ask a concise clarifying question for ONLY the missing fields, referencing `flight_params` to avoid re-asking for provided information. Example: If `departure_id` is provided but `arrival_id` and `outbound_date` are missing, ask: 'Please provide the destination airport and outbound date.'\n"
        "4) If all required fields are provided, confirm them with the user (e.g., 'Please confirm: traveling from {flight_params[departure_id]} to {flight_params[arrival_id]} on {flight_params[outbound_date]}, returning on {flight_params[return_date]}').\n"
        "5) After confirmation, ask for optional preferences (e.g., direct flights only, specific airlines, travel class) if not already provided. Use defaults if not specified: nonstop flights (`stops: '1'`), economy class (`travel_class: '1'`), 1 adult (`adults: 1`), no children/infants (`children: 0`, `infants_in_seat: 0`, `infants_on_lap: 0`), no airline preference.\n"
        "6) After preferences are collected, construct a payload for the `flights_finder` tool with STRUCTURED ARGUMENTS based on the Google Flights API parameters:\n"
        "   - The payload must be a dictionary with a single key `params` containing:\n"
        "     - `departure_id`: Airport code(s) or kgmid(s) (e.g., 'CDG' or 'CDG,ORY').\n"
        "     - `arrival_id`: Airport code(s) or kgmid(s) (e.g., 'AUS').\n"
        "     - `gl`: Set to 'in' for India (default).\n"
        "     - `hl`: Set to 'en' for English (default).\n"
        "     - `currency`: Set to 'INR' for Indian Rupees (default).\n"
        "     - `type`: '1' (round trip, default), '2' (one-way), or '3' (multi-city).\n"
        "     - `outbound_date`: YYYY-MM-DD (e.g., '2025-08-03').\n"
        "     - `return_date`: YYYY-MM-DD (e.g., '2025-08-09') for round trip; omit for one-way.\n"
        "     - `travel_class`: '1' (Economy, default), '2' (Premium economy), '3' (Business), '4' (First).\n"
        "     - `adults`: Default to 1.\n"
        "     - `children`: Default to 0.\n"
        "     - `infants_in_seat`: Default to 0.\n"
        "     - `infants_on_lap`: Default to 0.\n"
        "     - `stops`: '1' (Nonstop, default), '0' (Any), '2' (1 stop or fewer), '3' (2 stops or fewer).\n"
        "     - Optional: `include_airlines`, `exclude_airlines`, `max_price`, `outbound_times`, `return_times`, `max_duration`, `multi_city_json` (include only if specified).\n"
        "7) Return results succinctly, including:\n"
        "   - Prices in INR (e.g., '₹83,500 round-trip').\n"
        "   - Links to flight websites (e.g., `google_flights_url`).\n"
        "   - Airline logos (from `airline_logo`, if available).\n"
        "   - Key flight details (e.g., departure/arrival times, duration, stops, airline, flight number).\n"
        "   - If the tool returns JSON, keep JSON intact and summarize key details.\n"
        f"The current year is {CURRENT_YEAR}.\n\n"
        "IMPORTANT REACT FORMAT:\n"
        "- When reasoning, use 'Thought:'.\n"
        "- To use a tool, output exactly:\n"
        "  Action: <tool_name>\n"
        "  Action Input: <JSON arguments>\n"
        "  (Then wait for Observation.)\n"
        "- The Action Input must be a JSON dictionary, NOT a string. Example:\n"
        "  Correct: Action Input: {{\"params\": {{\"departure_id\": \"BOS\", \"arrival_id\": \"AUS\", ...}}}}\n"
        "  Incorrect: Action Input: '{{\"params\": {{\"departure_id\": \"BOS\", ...}}}}'\n"
        "- When replying to the user without calling a tool, output exactly:\n"
        "  Final Answer: <your message>\n\n"
        "Tools available: {tools}\n"
        "Tool names: {tool_names}\n\n"
        "Current flight parameters: {flight_params}\n"
        "Use a short scratchpad for your reasoning.\n"
        "Scratchpad:\n{agent_scratchpad}\n\n"
        "Conversation so far:\n{input}"
    )
)

# prompt = ChatPromptTemplate.from_template(
#     """You are a smart travel agency. Use the tools to look up information.
#     You are allowed to make multiple calls (either together or in sequence).
#     Only look up information when you are sure of what you want.
#     The current year is {CURRENT_YEAR}.
#     "Tools available: {tools}\n"
#     "Tool names: {tool_names}\n\n"
#     "Use a short scratchpad for your reasoning.\n"
#     "Scratchpad:\n{agent_scratchpad}\n\n"
#     If you need to look up some information before asking a follow up question, you are allowed to do that!
#     I want to have in your output links to hotels websites and flights websites (if possible).
#     I want to have as well the logo of the hotel and the logo of the airline company (if possible).
#     In your output always include the price of the flight and the price of the hotel and the currency as well (if possible).
#     for example for hotels-
#     Rate: $581 per night
#     Total: $3,488"""
# )
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=6,
    early_stopping_method="generate",
    return_intermediate_steps=True,
)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def agent_node(state: AgentState):
    # Initialize flight_params if not present
    if "flight_params" not in state:
        state["flight_params"] = {}

    conversation_text = "\n".join(
        f"User: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}"
        for m in state["messages"]
    )
    
    try:
        # Pass flight_params and conversation history to the agent
        result = agent_executor.invoke({
            "input": conversation_text,
            "flight_params": state["flight_params"],
            "tools": [t.name for t in tools],  # Pass tool names as strings
            "tool_names": tool_names,
            "agent_scratchpad": ""  # Initialize scratchpad (modify as needed)
        })
        
        content_text = str(result.get("output", ""))
        steps = result.get("intermediate_steps", []) or []
        
        # Log Action Input for debugging
        for step in steps:
            action, observation = step
            print(f"Debug - Action: {action.tool}")
            print(f"Debug - Action Input (Raw): {action.tool_input}")
            # Parse stringified JSON if necessary
            if isinstance(action.tool_input, str):
                try:
                    action.tool_input = json.loads(action.tool_input)
                    print(f"Debug - Action Input (Parsed): {action.tool_input}")
                except json.JSONDecodeError as e:
                    print(f"Debug - Failed to parse Action Input: {e}")
            print(f"Debug - Observation: {observation}")
            
            # Update flight_params based on Action Input
            if action.tool == "flights_finder" and isinstance(action.tool_input, dict):
                params = action.tool_input.get("params", {})
                state["flight_params"].update(params)
        
        # If the agent asks for clarification, update flight_params based on user response
        if "Please confirm" not in content_text and "Final Answer" in content_text:
            # Extract parameters from user input (simplified; enhance with NLP if needed)
            last_message = state["messages"][-1].content if state["messages"] else ""
            if "from" in last_message.lower() and "to" in last_message.lower():
                # Example: Extract parameters (customize based on your needs)
                if "departure_id" not in state["flight_params"]:
                    state["flight_params"]["departure_id"] = extract_airport(last_message, "from")
                if "arrival_id" not in state["flight_params"]:
                    state["flight_params"]["arrival_id"] = extract_airport(last_message, "to")
                if "on" in last_message.lower() and "outbound_date" not in state["flight_params"]:
                    state["flight_params"]["outbound_date"] = extract_date(last_message)
                if "return" in last_message.lower() and "return_date" not in state["flight_params"]:
                    state["flight_params"]["return_date"] = extract_date(last_message, "return")
        
        # Check for missing required fields
        required_fields = ["departure_id", "arrival_id", "outbound_date"]
        missing_fields = [f for f in required_fields if f not in state["flight_params"] or not state["flight_params"][f]]
        if missing_fields and "Please confirm" not in content_text:
            content_text = f"Final Answer: Please provide the following missing information: {', '.join(missing_fields)}."

        # If all required fields are present, ensure defaults for optional fields
        if not missing_fields:
            state["flight_params"].setdefault("gl", "in")
            state["flight_params"].setdefault("hl", "en")
            state["flight_params"].setdefault("currency", "INR")
            state["flight_params"].setdefault("type", "1")
            state["flight_params"].setdefault("travel_class", "1")
            state["flight_params"].setdefault("adults", 1)
            state["flight_params"].setdefault("children", 0)
            state["flight_params"].setdefault("infants_in_seat", 0)
            state["flight_params"].setdefault("infants_on_lap", 0)
            state["flight_params"].setdefault("stops", "1")

        if steps:
            last_obs = steps[-1][1]
            if isinstance(last_obs, dict) and "status" in last_obs and "response" in last_obs:
                content_text = json.dumps(last_obs)

        return {
            "messages": [AIMessage(content=content_text)],
            "flight_params": state["flight_params"]
        }
    except Exception as e:
        print(f"Error in agent_node: {e}")
        raise

# Helper functions to extract parameters (simplified; replace with robust NLP if needed)
def extract_airport(text: str, keyword: str) -> Optional[str]:
    # Example: Extract airport code from "from Boston" or "to Austin"
    words = text.lower().split()
    try:
        idx = words.index(keyword) + 1
        airport = words[idx].upper()  # Assume airport code like "BOS"
        return airport if len(airport) == 3 else None
    except (ValueError, IndexError):
        return None

def extract_date(text: str, keyword: str = None) -> Optional[str]:
    # Example: Extract date from "on August 3, 2025"
    import re
    pattern = r"\b(\d{4}-\d{2}-\d{2})\b"
    match = re.search(pattern, text)
    return match.group(1) if match else None

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

def _extract_thread_and_message(query: str):
    prefix = "THREAD_ID::"
    if query.startswith(prefix):
        try:
            _, rest = query.split(prefix, 1)
            thread_id, message = rest.split("::", 1)
            return thread_id.strip(), message.strip()
        except Exception:
            pass
    return "default-thread", query

def run_agent(query: str):
    thread_id, clean_message = _extract_thread_and_message(query)
    inputs = {"messages": [HumanMessage(content=clean_message)]}
    result = app.invoke(inputs, config={"configurable": {"thread_id": thread_id}})  # Use synchronous invoke
    return result["messages"][-1].content