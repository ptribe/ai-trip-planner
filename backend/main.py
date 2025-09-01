from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import time
import random
from datetime import datetime

# Load environment variables
load_dotenv()

# Arize and tracing imports
from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
try:
    from openinference.instrumentation.langgraph import LangGraphInstrumentor  # optional
    _HAS_LG_INSTR = True
except Exception:
    LangGraphInstrumentor = None  # type: ignore
    _HAS_LG_INSTR = False
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
from langchain.callbacks.base import BaseCallbackHandler
import json

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from openinference.instrumentation import using_prompt_template
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Import Airtable integration
from airtable_integration import airtable_logger

# Configure LiteLLM
import litellm
litellm.drop_params = True

# Global tracer
tracer = trace.get_tracer(__name__)

class PromptMetadataCallback(BaseCallbackHandler):
    """Attach prompt template and variables to the active LLM span."""

    def __init__(self, template: str, variables: Dict[str, Any], version: str = "v1"):
        self.template = template
        self.variables = variables
        self.version = version

    def on_llm_start(self, serialized, prompts, **kwargs):  # type: ignore[override]
        span = trace.get_current_span()
        try:
            if span and span.is_recording():
                span.set_attribute(SpanAttributes.PROMPT_TEMPLATE, self.template)
                span.set_attribute(SpanAttributes.PROMPT_TEMPLATE_VARIABLES, json.dumps(self.variables))
                span.set_attribute(SpanAttributes.PROMPT_TEMPLATE_VERSION, self.version)
        except Exception:
            pass

# Initialize LLM - use OpenAI or fallback to Groq
if os.getenv("OPENAI_API_KEY"):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=2000  # Increased for complete responses
    )
    print("✅ Using OpenAI GPT-3.5-turbo")
elif os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_KEY") != "your_groq_api_key_here":
    llm = ChatOpenAI(
        model="groq/llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        temperature=0.7
    )
    print("✅ Using Groq Llama3")
else:
    raise ValueError("No valid API key found. Please set OPENAI_API_KEY or GROQ_API_KEY in .env")

# Initialize search tools
search_tools = []
if os.getenv("TAVILY_API_KEY"):
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        search_tools.append(TavilySearchResults(max_results=5))
        print("✅ Tavily search tool initialized")
    except:
        print("⚠️ Tavily API key provided but initialization failed")

# Initialize Arize tracing
def setup_tracing():
    """Set up Arize tracing for the application"""
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        
        if not space_id or not api_key:
            print("⚠️ Arize credentials not configured. Tracing disabled.")
            return None
            
        # Register with Arize
        tracer_provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name="ai-trip-planner-with-tools-v2"
        )
        
        # Instrument LangGraph (if available), LangChain, and LiteLLM
        if _HAS_LG_INSTR:
            LangGraphInstrumentor().instrument(
                tracer_provider=tracer_provider
            )
        else:
            print("ℹ️ openinference-instrumentation-langgraph not installed; skipping LangGraph instrumentation.")

        # Keep LangChain instrumentation to generate LLM spans used by callbacks
        LangChainInstrumentor().instrument(
            tracer_provider=tracer_provider,
            include_tools=True,
            include_chains=True,
            include_agents=True
        )
        
        # Instrument LiteLLM for clients that go through LiteLLM
        LiteLLMInstrumentor().instrument(
            tracer_provider=tracer_provider,
            skip_dep_check=True
        )
        
        print("✅ Arize tracing initialized successfully")
        print(f"📊 Project: ai-trip-planner-with-tools-v2")
        print(f"🔗 Space ID: {space_id[:10]}...")
        print(f"🌐 View traces at: https://app.arize.com/")
        
        return tracer_provider
        
    except Exception as e:
        print(f"⚠️ Arize tracing setup failed: {str(e)}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    setup_tracing()
    yield

app = FastAPI(title="Trip Planner API with Tools", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TripRequest(BaseModel):
    destination: str
    duration: str
    budget: Optional[str] = None
    interests: Optional[str] = None
    travel_style: Optional[str] = None

class TripResponse(BaseModel):
    result: str
    tool_calls: List[Dict[str, Any]] = []

# ========== RESEARCH TOOLS ==========

@tool
def search_destination_info(destination: str) -> str:
    """Search for general destination information using web search.
    
    Args:
        destination: The destination to search for
    """
    if search_tools:
        search_result = search_tools[0].invoke(f"{destination} travel guide attractions weather culture")
        return f"Search results for {destination}: {str(search_result)[:500]}"
    return f"General info for {destination}: Popular tourist destination with varied attractions and cultural sites."

@tool
def get_destination_weather(destination: str) -> str:
    """Get weather information for the destination.
    
    Args:
        destination: The destination
    """
    time.sleep(random.uniform(0.5, 1.5))  # Simulate API call
    return f"Weather in {destination}: Average temperature 20-25°C, moderate rainfall. Best time: April-October."

@tool
def research_visa_requirements(destination: str) -> str:
    """Research visa requirements for the destination.
    
    Args:
        destination: The destination country
    """
    time.sleep(random.uniform(0.5, 1.5))  # Simulate API call
    countries = {"Japan": "Visa on arrival for US citizens (30 days), fee: $50",
                 "France": "No visa required for US citizens (90 days)",
                 "Thailand": "Visa on arrival (30 days), fee: $35"}
    return f"Visa for {destination}: {countries.get(destination, 'Check embassy requirements')}"

@tool
def find_local_events(destination: str) -> str:
    """Find local events and festivals happening at the destination.
    
    Args:
        destination: The destination
    """
    time.sleep(random.uniform(0.5, 1.5))  # Simulate API call
    events = ["Cultural Festival (weekends)", "Night Market (daily)", "Art Exhibition (various locations)"]
    return f"Events in {destination}: {', '.join(events)}"

# ========== BUDGET TOOLS ==========

@tool
def calculate_accommodation_cost(destination: str, duration: str, level: str = "mid-range") -> str:
    """Calculate accommodation costs for the trip.
    
    Args:
        destination: The destination
        duration: Trip duration
        level: Accommodation level (budget/mid-range/luxury)
    """
    costs = {"budget": "$30-50/night", "mid-range": "$80-150/night", "luxury": "$200-500/night"}
    return f"Accommodation in {destination} ({level}) for {duration}: {costs.get(level, costs['mid-range'])}"

@tool
def calculate_food_cost(destination: str, duration: str, style: str = "mixed") -> str:
    """Calculate food and dining costs.
    
    Args:
        destination: The destination
        duration: Trip duration
        style: Dining style (street_food/mixed/fine_dining)
    """
    costs = {"street_food": "$15-25/day", "mixed": "$40-60/day", "fine_dining": "$100-200/day"}
    return f"Food costs in {destination} ({style}) for {duration}: {costs.get(style, costs['mixed'])}"

@tool
def calculate_transport_cost(destination: str, duration: str, mode: str = "public") -> str:
    """Calculate transportation costs.
    
    Args:
        destination: The destination
        duration: Trip duration
        mode: Transport mode (public/taxi/rental_car)
    """
    costs = {"public": "$5-15/day", "taxi": "$30-50/day", "rental_car": "$40-80/day"}
    return f"Transport in {destination} ({mode}) for {duration}: {costs.get(mode, costs['public'])}"

@tool
def get_attraction_prices(destination: str, attractions: List[str] = None) -> str:
    """Get entrance fees for specific attractions.
    
    Args:
        destination: The destination
        attractions: List of attractions to check
    """
    if not attractions:
        attractions = ["Major Museum", "Historical Site", "Theme Park"]
    prices = [f"{attr}: ${random.randint(10, 50)}" for attr in attractions]
    return f"Attraction prices in {destination}: {', '.join(prices)}"

# ========== LOCAL EXPERIENCES TOOLS ==========

@tool
def find_local_restaurants(destination: str, cuisine_type: Optional[str] = None) -> str:
    """Find local restaurants and eateries.
    
    Args:
        destination: The destination
        cuisine_type: Type of cuisine (optional)
    """
    cuisine = cuisine_type or "local"
    return f"Restaurants in {destination} ({cuisine}): 1. Casa Local (authentic), 2. Street Food Market (budget), 3. Hidden Gem Bistro"

@tool
def find_cultural_activities(destination: str, interests: Optional[str] = None) -> str:
    """Find cultural activities and experiences.
    
    Args:
        destination: The destination
        interests: Specific interests (optional)
    """
    return f"Cultural activities in {destination}: Traditional dance show, Cooking class, Artisan workshop, Local festival"

@tool
def find_hidden_gems(destination: str, category: Optional[str] = None) -> str:
    """Find off-the-beaten-path attractions and hidden gems.
    
    Args:
        destination: The destination
        category: Category of hidden gems (optional)
    """
    return f"Hidden gems in {destination}: Secret garden cafe, Underground art gallery, Local's favorite viewpoint"

@tool
def get_local_customs(destination: str) -> str:
    """Get information about local customs and etiquette.
    
    Args:
        destination: The destination
    """
    return f"Customs in {destination}: Tipping 10-15%, Remove shoes indoors, Greet with handshake, Dress modestly at temples"

# ========== ITINERARY TOOLS ==========

@tool
def create_daily_schedule(destination: str, day_number: int, interests: List[str] = None) -> str:
    """Create a detailed daily schedule.
    
    Args:
        destination: The destination
        day_number: Which day of the trip
        interests: List of interests to incorporate
    """
    return f"Day {day_number} in {destination}: 9AM Breakfast, 10AM Museum visit, 1PM Local lunch, 3PM Walking tour, 7PM Dinner"

@tool
def calculate_travel_time(from_location: str, to_location: str, mode: str = "public") -> str:
    """Calculate travel time between locations.
    
    Args:
        from_location: Starting location
        to_location: Destination location
        mode: Transportation mode
    """
    time.sleep(random.uniform(0.5, 1.5))  # Simulate API call
    return f"Travel from {from_location} to {to_location} by {mode}: {random.randint(15, 60)} minutes"

@tool
def book_recommendations(destination: str, duration: str) -> str:
    """Get recommendations for what needs advance booking.
    
    Args:
        destination: The destination
        duration: Trip duration
    """
    return f"Must book in {destination}: Popular restaurants (2 weeks ahead), Tours (1 week ahead), Shows (3 days ahead)"

@tool
def create_packing_list(destination: str, duration: str, activities: List[str] = None) -> str:
    """Create a customized packing list.
    
    Args:
        destination: The destination
        duration: Trip duration
        activities: Planned activities
    """
    return f"Packing for {destination}: Weather-appropriate clothing, Comfortable walking shoes, Camera, Adapter, Sunscreen"

# Define the state for our graph
class TripPlannerState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    research_data: Optional[str]
    budget_data: Optional[str]
    local_data: Optional[str]
    final_result: Optional[str]
    tool_calls_made: Annotated[List[Dict[str, Any]], operator.add]  # Use operator.add for concurrent updates

# ========== AGENT NODES ==========

def research_agent(state: TripPlannerState) -> TripPlannerState:
    """Research agent that gathers destination information"""
    try:
        trip_req = state["trip_request"]
        destination = trip_req["destination"]

        print(f"🔍 Research agent starting for {destination}")

        # Research agent tools
        research_tools = [
            search_destination_info,
            get_destination_weather,
            research_visa_requirements,
            find_local_events
        ]

        # Create LLM with tools
        research_llm = llm.bind_tools(research_tools)

        # Prompt template + variables
        research_prompt_template = (
            "You are a destination research specialist.\n"
            "Research {destination} and gather essential travel information.\n"
            "Use 1-2 of your available tools to get key information about:\n"
            "- General destination overview\n"
            "- Weather conditions\n"
            "- Visa requirements\n"
            "- Local events\n\n"
            "Be selective and efficient - only call the most relevant tools."
        )
        research_vars = {"destination": destination}
        research_prompt = research_prompt_template.format(**research_vars)
        messages = [
            SystemMessage(content=research_prompt),
            HumanMessage(content=f"Research {destination} for a trip"),
        ]
        with using_prompt_template(template=research_prompt_template, variables=research_vars, version="v1"):
            response = research_llm.invoke(messages)

        # Track tool calls
        tool_calls = []
        result_messages = [response]

        # Process tool calls if any
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_calls.append({
                    "agent": "research",
                    "tool": tool_call["name"],
                    "args": tool_call.get("args", {})
                })

            # Execute tools
            tool_node = ToolNode(research_tools)
            tool_results = tool_node.invoke({"messages": [response]})
            result_messages.extend(tool_results["messages"])

        # Extract research data
        if tool_calls:
            research_data = "\n".join(
                [getattr(m, 'content', str(m)) for m in result_messages]
            )[:4000]
        else:
            research_data = getattr(response, 'content', "Research completed")

        print(f"✅ Research agent completed with {len(tool_calls)} tool calls")

        return {
            "messages": result_messages,
            "research_data": research_data,
            "tool_calls_made": tool_calls  # Just return the new tool calls
        }

    except Exception as e:
        print(f"❌ Research agent error: {str(e)}")
        return {
            "messages": [HumanMessage(content=f"Research failed: {str(e)}")],
            "research_data": f"Research failed: {str(e)}"
        }

def budget_agent(state: TripPlannerState) -> TripPlannerState:
    """Budget agent that analyzes costs"""
    try:
        trip_req = state["trip_request"]
        destination = trip_req["destination"]
        duration = trip_req["duration"]
        budget = trip_req.get("budget", "flexible")

        print(f"💰 Budget agent starting for {destination}")

        # Budget agent tools
        budget_tools = [
            calculate_accommodation_cost,
            calculate_food_cost,
            calculate_transport_cost,
            get_attraction_prices
        ]

        # Create LLM with tools
        budget_llm = llm.bind_tools(budget_tools)

        # Prompt template + variables
        budget_prompt_template = (
            "You are a travel budget analyst.\n"
            "Analyze costs for a {duration} trip to {destination} with budget: {budget}.\n"
            "Use 1-2 of your available tools to calculate:\n"
            "- Accommodation costs\n"
            "- Food expenses\n"
            "- Transportation costs\n"
            "- Attraction fees\n\n"
            "Focus on the most important cost categories for this trip."
        )
        budget_vars = {"duration": duration, "destination": destination, "budget": budget}
        budget_prompt = budget_prompt_template.format(**budget_vars)
        messages = [
            SystemMessage(content=budget_prompt),
            HumanMessage(content=f"Calculate budget for {destination} trip"),
        ]
        with using_prompt_template(template=budget_prompt_template, variables=budget_vars, version="v1"):
            response = budget_llm.invoke(messages)

        # Track tool calls
        tool_calls = []
        result_messages = [response]

        # Process tool calls if any
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_calls.append({
                    "agent": "budget",
                    "tool": tool_call["name"],
                    "args": tool_call.get("args", {})
                })

            # Execute tools
            tool_node = ToolNode(budget_tools)
            tool_results = tool_node.invoke({"messages": [response]})
            result_messages.extend(tool_results["messages"])

        # Extract budget data
        if tool_calls:
            budget_data = "\n".join(
                [getattr(m, 'content', str(m)) for m in result_messages]
            )[:4000]
        else:
            budget_data = getattr(response, 'content', "Budget analysis completed")

        print(f"✅ Budget agent completed with {len(tool_calls)} tool calls")

        return {
            "messages": result_messages,
            "budget_data": budget_data,
            "tool_calls_made": tool_calls  # Just return the new tool calls
        }

    except Exception as e:
        print(f"❌ Budget agent error: {str(e)}")
        return {
            "messages": [HumanMessage(content=f"Budget analysis failed: {str(e)}")],
            "budget_data": f"Budget analysis failed: {str(e)}"
        }

def local_experiences_agent(state: TripPlannerState) -> TripPlannerState:
    """Local experiences agent that finds authentic activities"""
    try:
        trip_req = state["trip_request"]
        destination = trip_req["destination"]
        interests = trip_req.get("interests", "general exploration")

        print(f"🍽️ Local experiences agent starting for {destination}")

        # Local experiences tools
        local_tools = [
            find_local_restaurants,
            find_cultural_activities,
            find_hidden_gems,
            get_local_customs
        ]

        # Create LLM with tools
        local_llm = llm.bind_tools(local_tools)

        # Prompt template + variables
        local_prompt_template = (
            "You are a local experiences curator.\n"
            "Find authentic experiences in {destination} for someone interested in: {interests}.\n"
            "Use 1-2 of your available tools to discover:\n"
            "- Local restaurants\n"
            "- Cultural activities\n"
            "- Hidden gems\n"
            "- Local customs\n\n"
            "Focus on unique, authentic experiences that match the traveler's interests."
        )
        local_vars = {"destination": destination, "interests": interests}
        local_prompt = local_prompt_template.format(**local_vars)
        messages = [
            SystemMessage(content=local_prompt),
            HumanMessage(content=f"Find local experiences in {destination}"),
        ]
        with using_prompt_template(template=local_prompt_template, variables=local_vars, version="v1"):
            response = local_llm.invoke(messages)

        # Track tool calls
        tool_calls = []
        result_messages = [response]

        # Process tool calls if any
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_calls.append({
                    "agent": "local_experiences",
                    "tool": tool_call["name"],
                    "args": tool_call.get("args", {})
                })

            # Execute tools
            tool_node = ToolNode(local_tools)
            tool_results = tool_node.invoke({"messages": [response]})
            result_messages.extend(tool_results["messages"])

        # Extract local data
        if tool_calls:
            local_data = "\n".join(
                [getattr(m, 'content', str(m)) for m in result_messages]
            )[:4000]
        else:
            local_data = getattr(response, 'content', "Local experiences found")

        print(f"✅ Local experiences agent completed with {len(tool_calls)} tool calls")

        return {
            "messages": result_messages,
            "local_data": local_data,
            "tool_calls_made": tool_calls  # Just return the new tool calls
        }

    except Exception as e:
        print(f"❌ Local experiences agent error: {str(e)}")
        return {
            "messages": [HumanMessage(content=f"Local experiences failed: {str(e)}")],
            "local_data": f"Local experiences failed: {str(e)}"
        }

def itinerary_agent(state: TripPlannerState) -> TripPlannerState:
    """Itinerary agent that creates the final trip plan (single LLM synthesis, no tools)."""
    try:
        trip_req = state["trip_request"]
        destination = trip_req["destination"]
        duration = trip_req["duration"]
        travel_style = trip_req.get("travel_style", "standard")

        print(f"📅 Itinerary agent starting for {destination}")

        # Single-shot synthesis LLM (no tools)
        itinerary_llm = llm

        # Get data from other agents
        research_data = state.get("research_data", "")
        budget_data = state.get("budget_data", "")
        local_data = state.get("local_data", "")

        # Prompt template + variables (synthesis only)
        itinerary_prompt_template = (
            "You are a trip itinerary planner.\n"
            "Create a {duration} itinerary for {destination} ({travel_style} style).\n\n"
            "Available information:\n"
            "Research: {research_excerpt}\n"
            "Budget: {budget_excerpt}\n"
            "Local: {local_excerpt}\n\n"
            "Focus on creating a practical, enjoyable itinerary with clear sections."
        )
        itinerary_vars = {
            "duration": duration,
            "destination": destination,
            "travel_style": travel_style,
            "research_excerpt": (research_data or "")[:300],
            "budget_excerpt": (budget_data or "")[:300],
            "local_excerpt": (local_data or "")[:300],
        }
        
        itinerary_prompt = itinerary_prompt_template.format(**itinerary_vars)
        messages = [
            SystemMessage(content=itinerary_prompt),
            HumanMessage(content=f"Create itinerary for {destination}"),
        ]
        with using_prompt_template(template=itinerary_prompt_template, variables=itinerary_vars, version="v1"):
            response = itinerary_llm.invoke(messages)

        # Single-shot synthesis
        tool_calls = []
        result_messages = []

        # Extract final itinerary
        final_result = getattr(response, 'content', '') or (
            f"Trip plan for {destination} ({duration})."  # minimal fallback
        )

        print("✅ Itinerary agent completed with 0 tool calls (synthesis only)")

        return {
            "messages": result_messages,
            "final_result": final_result,
            "tool_calls_made": tool_calls  # Just return the new tool calls
        }

    except Exception as e:
        print(f"❌ Itinerary agent error: {str(e)}")
        return {
            "messages": [HumanMessage(content=f"Itinerary creation failed: {str(e)}")],
            "final_result": f"Itinerary creation failed: {str(e)}"
        }

# Build the graph
def create_trip_planning_graph():
    """Create and compile the trip planning graph with agent-based architecture"""
    
    # Create the state graph
    workflow = StateGraph(TripPlannerState)
    
    # Add agent nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("budget", budget_agent)
    workflow.add_node("local_experiences", local_experiences_agent)
    workflow.add_node("itinerary", itinerary_agent)
    
    # Sequential execution to ensure itinerary sees all upstream results
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "budget")
    workflow.add_edge("budget", "local_experiences")
    workflow.add_edge("local_experiences", "itinerary")
    
    # Itinerary ends the workflow
    workflow.add_edge("itinerary", END)
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# API Routes
@app.get("/")
async def root():
    return {"message": "Trip Planner API with Agent-Based Tools is running!"}

@app.post("/plan-trip", response_model=TripResponse)
async def plan_trip(trip_request: TripRequest):
    """Plan a trip using agent-based workflow with multiple tool calls per agent"""
    try:
        # Log to Airtable if available
        if airtable_logger:
            airtable_logger.log_request(trip_request.model_dump())

        # Create the graph
        graph = create_trip_planning_graph()

        # Prepare initial state
        initial_state = {
            "messages": [],
            "trip_request": trip_request.model_dump(),
            "research_data": None,
            "budget_data": None,
            "local_data": None,
            "final_result": None,
            "tool_calls_made": []
        }

        # Execute the workflow
        config = {
            "configurable": {
                "thread_id": f"trip_{trip_request.destination}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        }

        print(f"🚀 Starting trip planning for {trip_request.destination} ({trip_request.duration})")
        start_time = time.time()

        # Invoke the graph
        output = graph.invoke(initial_state, config)

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        print(f"✅ Trip planning completed in {latency_ms:.0f}ms with {len(output.get('tool_calls_made', []))} tool calls")

        # Log result to Airtable
        if airtable_logger and output:
            airtable_logger.log_response(
                trip_request.model_dump(),
                output.get("final_result", "No result"),
                output.get("tool_calls_made", [])
            )

        # Return the final result
        return TripResponse(
            result=output.get("final_result", "Trip planning completed."),
            tool_calls=output.get("tool_calls_made", [])
        )

    except Exception as e:
        print(f"❌ Trip planning error: {str(e)}")
        import traceback
        traceback.print_exc()

        # Log error to Airtable
        if airtable_logger:
            airtable_logger.log_error(trip_request.model_dump(), str(e))

        raise HTTPException(status_code=500, detail=f"Trip planning failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "trip-planner-with-tools-fixed"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
