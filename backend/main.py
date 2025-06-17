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
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
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

# Configure LiteLLM
import litellm
litellm.set_verbose = True  # Enable debug logging for LiteLLM
litellm.drop_params = True  # Drop unsupported parameters automatically

# Global tracer provider to ensure it's available across the application
tracer_provider = None

# Initialize Arize tracing
def setup_tracing():
    global tracer_provider
    try:
        # Check if required environment variables are set
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        
        if not space_id or not api_key or space_id == "your_arize_space_id_here" or api_key == "your_arize_api_key_here":
            print("⚠️ Arize credentials not configured properly.")
            print("📝 Please set ARIZE_SPACE_ID and ARIZE_API_KEY environment variables.")
            print("📝 Copy backend/env_example.txt to backend/.env and update with your credentials.")
            return None
            
        tracer_provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name="trip-planner"
        )
        
        # Instrument all relevant components
        # OpenAI instrumentation (for ChatOpenAI)
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        
        # LangChain instrumentation (for tools and chains)
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        
        # LiteLLM instrumentation (if using LiteLLM directly)
        LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
        
        print("✅ Arize tracing initialized successfully")
        print(f"📊 Project: trip-planner")
        print(f"🔗 Space ID: {space_id[:8]}...")
        
        return tracer_provider
        
    except Exception as e:
        print(f"⚠️ Arize tracing setup failed: {str(e)}")
        print("📝 Continuing without tracing - check your ARIZE_SPACE_ID and ARIZE_API_KEY")
        print("📝 Also ensure you have the latest version of openinference packages")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup tracing before anything else
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

# Initialize the LLM - Using Groq for faster inference
# Note: This should be initialized after instrumentation setup
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY", os.getenv("OPENAI_API_KEY")),
    model="llama3-8b-8192",  # Fast and high-quality Groq model
    temperature=0,
    max_tokens=2000,
    timeout=30
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
        
        prompt_template = """Based on these search results about {destination}:
        {search_results}
        
        Provide comprehensive destination research for a {duration} trip including:
        - Weather conditions and best time to visit
        - Top attractions and activities
        - Cultural considerations and customs
        - Transportation options
        - Safety information
        - Seasonal factors and considerations
        - Local customs and etiquette
        - Essential travel tips"""
        
        prompt_template_variables = {
            "destination": destination,
            "duration": duration,
            "search_results": str(search_results)
        }
    else:
        prompt_template = """Provide comprehensive destination research for {destination} for a {duration} trip including:
        - Weather conditions and best time to visit  
        - Top attractions and activities
        - Cultural considerations and customs
        - Transportation options
        - Safety information
        - Seasonal factors
        - Local customs and etiquette
        - Essential travel tips
        
        Note: Using general knowledge as live search is unavailable."""
        
        prompt_template_variables = {
            "destination": destination,
            "duration": duration
        }
    
    with using_prompt_template(
        template=prompt_template,
        variables=prompt_template_variables,
        version="research-v1.0",
    ):
        formatted_prompt = prompt_template.format(**prompt_template_variables)
        response = llm.invoke([SystemMessage(content=formatted_prompt)])
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
    
    prompt_template = """Analyze budget requirements for a {duration} trip to {destination}.
    Target budget: {budget}
    
    Include detailed breakdown of:
    - Accommodation costs (budget, mid-range, luxury options)
    - Transportation (flights, local transport)
    - Food and dining expenses
    - Activities and attraction costs
    - Miscellaneous expenses
    - Money-saving tips and strategies
    - Total estimated costs by budget tier
    - Seasonal pricing considerations
    - Value-for-money recommendations"""
    
    prompt_template_variables = {
        "destination": destination,
        "duration": duration,
        "budget": budget_text
    }
    
    with using_prompt_template(
        template=prompt_template,
        variables=prompt_template_variables,
        version="budget-v1.0",
    ):
        formatted_prompt = prompt_template.format(**prompt_template_variables)
        response = llm.invoke([SystemMessage(content=formatted_prompt)])
    return response.content

@tool
def curate_local_experiences(destination: str, interests: str = None) -> str:
    """Curate authentic local experiences and hidden gems.
    
    Args:
        destination: The destination
        interests: Traveler interests (optional)
    """
    interests_text = interests or "general exploration and cultural immersion"
    
    prompt_template = """Curate authentic local experiences for {destination} focusing on traveler interests: {interests}
    
    Include recommendations for:
    - Hidden gem restaurants and authentic cuisine spots (avoid tourist traps)
    - Cultural activities and local events
    - Off-the-beaten-path locations and experiences
    - Traditional markets and unique shopping areas
    - Community experiences and local interactions
    - Activities that match the specified interests
    - Local artisan workshops and craft experiences
    - Traditional ceremonies or cultural events
    - Tips for respectful cultural engagement and etiquette"""
    
    prompt_template_variables = {
        "destination": destination,
        "interests": interests_text
    }
    
    with using_prompt_template(
        template=prompt_template,
        variables=prompt_template_variables,
        version="local-v1.0",
    ):
        formatted_prompt = prompt_template.format(**prompt_template_variables)
        response = llm.invoke([SystemMessage(content=formatted_prompt)])
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
    
    prompt_template = """Create a comprehensive day-by-day itinerary for {destination} lasting {duration}.
    Travel style: {travel_style}
    
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
    - Transportation suggestions between locations
    - Estimated daily costs and budget considerations
    - Practical tips and logistical considerations
    - Backup plans for weather or closure situations
    - Balance of must-see attractions and leisure time
    
    Format as a detailed day-by-day plan with clear structure and timing.
    EXTREMELY IMPORTANT: DO NOT EXCEED 100 words"""
    
    prompt_template_variables = {
        "destination": destination,
        "duration": duration,
        "travel_style": style_text,
        "research": research,
        "budget_info": budget_info,
        "local_info": local_info
    }
    
    with using_prompt_template(
        template=prompt_template,
        variables=prompt_template_variables,
        version="itinerary-v1.0",
    ):
        formatted_prompt = prompt_template.format(**prompt_template_variables)
        response = llm.invoke([SystemMessage(content=formatted_prompt)])
    return response.content

# Enhanced state to track parallel data
class EfficientTripPlannerState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    research_data: Optional[str]
    budget_data: Optional[str]
    local_data: Optional[str]
    final_result: Optional[str]

# Define more efficient nodes for parallel execution
def research_node(state: EfficientTripPlannerState) -> EfficientTripPlannerState:
    """Research destination in parallel"""
    try:
        trip_req = state["trip_request"]
        print(f"🔍 Starting research for {trip_req.get('destination', 'Unknown')}")
        
        research_result = research_destination.invoke({
            "destination": trip_req["destination"], 
            "duration": trip_req["duration"]
        })
        
        print(f"✅ Research completed for {trip_req.get('destination', 'Unknown')}")
        return {
            "messages": [HumanMessage(content=f"Research completed: {research_result}")],
            "research_data": research_result
        }
    except Exception as e:
        print(f"❌ Research node error: {str(e)}")
        return {
            "messages": [HumanMessage(content=f"Research failed: {str(e)}")],
            "research_data": f"Research failed: {str(e)}"
        }

def budget_node(state: EfficientTripPlannerState) -> EfficientTripPlannerState:
    """Analyze budget in parallel"""
    try:
        trip_req = state["trip_request"]
        print(f"💰 Starting budget analysis for {trip_req.get('destination', 'Unknown')}")
        
        budget_result = analyze_budget.invoke({
            "destination": trip_req["destination"], 
            "duration": trip_req["duration"], 
            "budget": trip_req.get("budget")
        })
        
        print(f"✅ Budget analysis completed for {trip_req.get('destination', 'Unknown')}")
        return {
            "messages": [HumanMessage(content=f"Budget analysis completed: {budget_result}")],
            "budget_data": budget_result
        }
    except Exception as e:
        print(f"❌ Budget node error: {str(e)}")
        return {
            "messages": [HumanMessage(content=f"Budget analysis failed: {str(e)}")],
            "budget_data": f"Budget analysis failed: {str(e)}"
        }

def local_experiences_node(state: EfficientTripPlannerState) -> EfficientTripPlannerState:
    """Curate local experiences in parallel"""
    try:
        trip_req = state["trip_request"]
        print(f"🍽️ Starting local experiences curation for {trip_req.get('destination', 'Unknown')}")
        
        local_result = curate_local_experiences.invoke({
            "destination": trip_req["destination"], 
            "interests": trip_req.get("interests")
        })
        
        print(f"✅ Local experiences completed for {trip_req.get('destination', 'Unknown')}")
        return {
            "messages": [HumanMessage(content=f"Local experiences curated: {local_result}")],
            "local_data": local_result
        }
    except Exception as e:
        print(f"❌ Local experiences node error: {str(e)}")
        return {
            "messages": [HumanMessage(content=f"Local experiences failed: {str(e)}")],
            "local_data": f"Local experiences failed: {str(e)}"
        }

def itinerary_node(state: EfficientTripPlannerState) -> EfficientTripPlannerState:
    """Create final itinerary using all gathered data"""
    try:
        trip_req = state["trip_request"]
        print(f"📅 Starting itinerary creation for {trip_req.get('destination', 'Unknown')}")
        
        # Get data from previous nodes
        research_data = state.get("research_data", "")
        budget_data = state.get("budget_data", "")
        local_data = state.get("local_data", "")
        
        print(f"📊 Data available - Research: {len(research_data) if research_data else 0} chars, Budget: {len(budget_data) if budget_data else 0} chars, Local: {len(local_data) if local_data else 0} chars")
        
        itinerary_result = create_itinerary.invoke({
            "destination": trip_req["destination"],
            "duration": trip_req["duration"],
            "research": research_data,
            "budget_info": budget_data,
            "local_info": local_data,
            "travel_style": trip_req.get("travel_style")
        })
        
        print(f"✅ Itinerary creation completed for {trip_req.get('destination', 'Unknown')}")
        return {
            "messages": [HumanMessage(content=itinerary_result)],
            "final_result": itinerary_result
        }
    except Exception as e:
        print(f"❌ Itinerary node error: {str(e)}")
        return {
            "messages": [HumanMessage(content=f"Itinerary creation failed: {str(e)}")],
            "final_result": f"Itinerary creation failed: {str(e)}"
        }

# Build the optimized graph with parallel execution
def create_efficient_trip_planning_graph():
    """Create and compile the optimized trip planning graph with parallel execution"""
    
    # Create the state graph
    workflow = StateGraph(EfficientTripPlannerState)
    
    # Add parallel processing nodes
    workflow.add_node("research", research_node)
    workflow.add_node("budget", budget_node)
    workflow.add_node("local_experiences", local_experiences_node)
    workflow.add_node("itinerary", itinerary_node)
    
    # Start all research tasks in parallel
    workflow.add_edge(START, "research")
    workflow.add_edge(START, "budget")
    workflow.add_edge(START, "local_experiences")
    
    # All parallel tasks feed into itinerary creation
    workflow.add_edge("research", "itinerary")
    workflow.add_edge("budget", "itinerary")
    workflow.add_edge("local_experiences", "itinerary")
    
    # Itinerary is the final step
    workflow.add_edge("itinerary", END)
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# API Routes
@app.get("/")
async def root():
    return {"message": "Trip Planner API is running with simplified LangGraph!"}


@app.post("/plan-trip", response_model=TripResponse)
async def plan_trip(trip_request: TripRequest):
    """Plan a trip using optimized parallel LangGraph workflow"""
    try:
        # Create the efficient graph
        graph = create_efficient_trip_planning_graph()
        
        # Prepare initial state with the new structure
        initial_state = {
            "messages": [],
            "trip_request": trip_request.model_dump(),
            "research_data": None,
            "budget_data": None,
            "local_data": None,
            "final_result": None
        }
        
        # Execute the workflow with parallel processing
        config = {"configurable": {"thread_id": f"trip_{trip_request.destination.replace(' ', '_')}_{trip_request.duration.replace(' ', '_')}"}}
        
        print(f"🚀 Starting trip planning for {trip_request.destination} ({trip_request.duration})")
        
        output = graph.invoke(initial_state, config)
        
        print(f"✅ Trip planning completed. Output keys: {list(output.keys()) if output else 'None'}")
        
        # Return the final result
        if output and output.get("final_result"):
            return TripResponse(result=output.get("final_result"))
        elif output and output.get("messages") and len(output.get("messages")) > 0:
            # Fallback to last message if final_result is not available
            last_message = output.get("messages")[-1]
            content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            return TripResponse(result=content)
        
        return TripResponse(result="Trip planning completed but no detailed results available.")
        
    except Exception as e:
        print(f"❌ Trip planning error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Trip planning failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "trip-planner-backend-simplified"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
