from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import asyncio

# Load environment variables from .env file
load_dotenv()

# Arize and tracing imports
from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation import using_prompt_template

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize Arize tracing
def setup_tracing():
    tracer_provider = register(
        space_id=os.getenv("ARIZE_SPACE_ID", "your-space-id"),
        api_key=os.getenv("ARIZE_API_KEY", "your-arize-api-key"),
        project_name="trip-planner"
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_tracing()
    yield

app = FastAPI(title="Trip Planner API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TripRequest(BaseModel):
    destination: str
    duration: str
    budget: Optional[str] = None
    interests: Optional[str] = None
    travel_style: Optional[str] = None

class TripResponse(BaseModel):
    result: str

# Define the state for our graph
class TripPlannerState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    final_result: Optional[str]

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize search tool if available
search_tools = []
if os.getenv("TAVILY_API_KEY"):
    search_tools.append(TavilySearchResults(max_results=5))

# Define trip planning tools
@tool
def research_destination(destination: str, duration: str) -> str:
    """Research a destination comprehensively for trip planning.
    
    Args:
        destination: The destination to research
        duration: Duration of the trip
    """
    # Use search tool if available, otherwise use LLM knowledge
    if search_tools:
        search_tool = search_tools[0]
        search_results = search_tool.invoke(f"{destination} travel guide {duration} trip attractions weather")
        
        prompt = f"""
        Based on these search results about {destination}:
        {search_results}
        
        Provide comprehensive destination research including:
        - Weather conditions and best time to visit
        - Top attractions and activities
        - Cultural considerations and customs
        - Transportation options
        - Safety information
        - Seasonal factors for a {duration} trip
        """
    else:
        prompt = f"""
        Provide comprehensive destination research for {destination} for a {duration} trip including:
        - Weather conditions and best time to visit  
        - Top attractions and activities
        - Cultural considerations and customs
        - Transportation options
        - Safety information
        - Seasonal factors
        Note: Using general knowledge as live search is unavailable.
        """
    
    prompt_template_variables = {
        "destination": destination,
        "duration": duration,
        "search_results": search_results if search_tools else "Not available"
    }
    
    with using_prompt_template(
        template=prompt,
        variables=prompt_template_variables,
        version="1.0",
    ):
        response = llm.invoke([SystemMessage(content=prompt)])
    return response.content

@tool
def analyze_budget(destination: str, duration: str, budget: str = None) -> str:
    """Analyze budget requirements for a trip.
    
    Args:
        destination: The destination
        duration: Duration of the trip
        budget: Target budget (optional)
    """
    budget_text = budget or "provide options for different budget levels"
    
    prompt = f"""
    Analyze budget requirements for a {duration} trip to {destination}.
    Target budget: {budget_text}
    
    Include detailed breakdown of:
    - Accommodation costs (budget, mid-range, luxury options)
    - Transportation (flights, local transport)
    - Food and dining expenses
    - Activities and attraction costs
    - Miscellaneous expenses
    - Money-saving tips and strategies
    - Total estimated costs by budget tier
    """
    
    prompt_template_variables = {
        "destination": destination,
        "duration": duration,
        "budget": budget_text
    }
    
    with using_prompt_template(
        template=prompt,
        variables=prompt_template_variables,
        version="1.0",
    ):
        response = llm.invoke([SystemMessage(content=prompt)])
    return response.content

@tool
def curate_local_experiences(destination: str, interests: str = None) -> str:
    """Curate authentic local experiences and hidden gems.
    
    Args:
        destination: The destination
        interests: Traveler interests (optional)
    """
    interests_text = interests or "general exploration and cultural immersion"
    
    prompt = f"""
    Curate authentic local experiences for {destination} focusing on traveler interests: {interests_text}
    
    Include:
    - Local restaurants and authentic cuisine spots (avoid tourist traps)
    - Cultural activities and events
    - Hidden gems and off-the-beaten-path locations
    - Local markets and shopping areas
    - Community experiences and interactions
    - Activities that match the specified interests
    - Tips for respectful cultural engagement
    """
    
    prompt_template_variables = {
        "destination": destination,
        "interests": interests_text
    }
    
    with using_prompt_template(
        template=prompt,
        variables=prompt_template_variables,
        version="1.0",
    ):
        response = llm.invoke([SystemMessage(content=prompt)])
    return response.content

@tool
def create_itinerary(destination: str, duration: str, research: str, budget_info: str, local_info: str, travel_style: str = None) -> str:
    """Create a comprehensive day-by-day itinerary.
    
    Args:
        destination: The destination
        duration: Duration of the trip
        research: Destination research information
        budget_info: Budget analysis information
        local_info: Local experiences information
        travel_style: Travel style preferences (optional)
    """
    style_text = travel_style or "Standard"
    
    prompt = f"""
    Create a comprehensive day-by-day itinerary for {destination} lasting {duration}.
    Travel style: {style_text}
    
    Base the itinerary on this information:
    
    DESTINATION RESEARCH:
    {research}
    
    BUDGET ANALYSIS:
    {budget_info}
    
    LOCAL EXPERIENCES:
    {local_info}
    
    Create a balanced itinerary that includes:
    - Specific daily schedules with timings
    - Mix of popular attractions and local experiences
    - Restaurant recommendations for each day
    - Transportation suggestions
    - Estimated daily costs
    - Practical tips and considerations
    - Backup plans for weather/closures
    
    Format as a detailed day-by-day plan with clear structure.
    """
    
    prompt_template_variables = {
        "destination": destination,
        "duration": duration,
        "travel_style": style_text,
        "research": research,
        "budget_info": budget_info,
        "local_info": local_info
    }
    
    with using_prompt_template(
        template=prompt,
        variables=prompt_template_variables,
        version="1.0",
    ):
        response = llm.invoke([SystemMessage(content=prompt)])
    return response.content

# Define the supervisor node
def supervisor_node(state: TripPlannerState) -> TripPlannerState:
    """Supervisor that uses tools to plan trips"""
    
    trip_req = state["trip_request"]
    
    supervisor_prompt = f"""
    You are a comprehensive trip planning assistant. Plan a {trip_req['duration']} trip to {trip_req['destination']}.
    
    Trip requirements:
    - Destination: {trip_req['destination']}
    - Duration: {trip_req['duration']}
    - Budget: {trip_req.get('budget', 'Flexible')}
    - Interests: {trip_req.get('interests', 'General sightseeing')}
    - Travel style: {trip_req.get('travel_style', 'Standard')}
    
    Use the available tools to plan the trip where necessary.
    """
    
    prompt_template_variables = {
        "destination": trip_req.get("destination", ""),
        "duration": trip_req.get("duration", ""),
        "budget": trip_req.get("budget", "Flexible"),
        "interests": trip_req.get("interests", "General sightseeing"),
        "travel_style": trip_req.get("travel_style", "Standard"),
    }
    
    messages = [SystemMessage(content=supervisor_prompt)]
    messages.extend(state.get("messages", []))
    
    # Bind all tools to the LLM
    all_tools = [research_destination, analyze_budget, curate_local_experiences, create_itinerary] + search_tools
    supervisor_llm = llm.bind_tools(all_tools)
    
    with using_prompt_template(
        template=supervisor_prompt,
        variables=prompt_template_variables,
        version="1.0",
    ):
        response = supervisor_llm.invoke(messages)
    
    return {
        "messages": [response]
    }


# Check if supervisor should call tools or finish
def should_continue(state: TripPlannerState) -> str:
    """Determine if supervisor should call tools or finish"""
    messages = state.get("messages", [])
    if not messages:
        return "tools"
    
    last_message = messages[-1]
    
    # If last message has tool calls, go to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # If we have a comprehensive response, we're done
    if hasattr(last_message, 'content') and last_message.content:
        if "day-by-day" in last_message.content.lower() or "itinerary" in last_message.content.lower():
            return END
    
    # Otherwise continue with tools
    return END


# Build the simplified graph
def create_trip_planning_graph():
    """Create and compile the simplified trip planning graph"""
    
    # Create the state graph
    workflow = StateGraph(TripPlannerState)
    
    # Add nodes - just supervisor and tools
    workflow.add_node("supervisor", supervisor_node)
    
    # Create tool node with all tools
    all_tools = [research_destination, analyze_budget, curate_local_experiences, create_itinerary] + search_tools
    # all_tools = [create_itinerary]
    tool_node = ToolNode(all_tools)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "supervisor")
    
    # Conditional edge from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # Tools always go back to supervisor
    workflow.add_edge("tools", "supervisor")
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# API Routes
@app.get("/")
async def root():
    return {"message": "Trip Planner API is running with simplified LangGraph!"}


@app.post("/plan-trip", response_model=TripResponse)
async def plan_trip(trip_request: TripRequest):
    """Plan a trip using simplified LangGraph workflow"""
    try:
        # Create the graph
        graph = create_trip_planning_graph()
        
        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content=f"Plan a comprehensive trip to {trip_request.destination} for {trip_request.duration}")],
            "trip_request": trip_request.dict(),
            "final_result": None
        }
        
        # Execute the workflow
        config = {"configurable": {"thread_id": "trip_planning_session"}}
        
        output = graph.invoke(initial_state, config)
        if output:
            return TripResponse(result=output.get("messages")[-1].content)
        
        return TripResponse(result="Trip planning completed but no detailed results available.")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "trip-planner-backend-simplified"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
