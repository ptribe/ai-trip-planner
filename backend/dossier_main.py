"""
Intelligence Dossier Agent System
Multi-agent AI system for generating comprehensive intelligence dossiers.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Arize AX Observability via OpenInference
try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template, using_metadata, using_attributes
    from opentelemetry import trace as trace_api
    from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
    _TRACING = True
except Exception:
    def using_prompt_template(**kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_metadata(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_attributes(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()

    class SpanAttributes:  # type: ignore
        OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
        INPUT_VALUE = "input.value"
        OUTPUT_VALUE = "output.value"

    class OpenInferenceSpanKindValues:  # type: ignore
        AGENT = "AGENT"
        CHAIN = "CHAIN"
        TOOL = "TOOL"
        LLM = "LLM"

    _TRACING = False

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import httpx


# Pydantic models for API
class DossierRequest(BaseModel):
    location: str
    mission_duration: Optional[str] = None
    mission_objectives: Optional[str] = None
    risk_tolerance: Optional[str] = "medium"
    focus_areas: Optional[List[str]] = None  # Optional: cultural, economic, political, security, operational, events
    # Observability fields
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class DossierResponse(BaseModel):
    executive_summary: str
    cultural_intel: str
    economic_intel: str
    political_intel: str
    security_intel: str
    operational_intel: str
    events_intel: str
    poi_profiles: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []


# Initialize LLM
def _init_llm():
    if os.getenv("TEST_MODE"):
        class _Fake:
            def bind_tools(self, tools):
                return self
            def invoke(self, messages):
                class _Msg:
                    content = "Test dossier content"
                    tool_calls: List[Dict[str, Any]] = []
                return _Msg()
        return _Fake()

    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=2000)
    elif os.getenv("OPENROUTER_API_KEY"):
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.7,
            max_tokens=2000,
        )
    else:
        raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")


llm = _init_llm()


# Search timeout
SEARCH_TIMEOUT = 10.0


def _compact(text: str, limit: int = 300) -> str:
    """Compact text to maximum length."""
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    truncated = cleaned[:limit]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(",.;- ")


def _search_api(query: str) -> Optional[str]:
    """Search the web using Tavily or SerpAPI if configured."""
    query = query.strip()
    if not query:
        return None

    # Try Tavily first
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": query,
                        "max_results": 5,
                        "search_depth": "advanced",
                        "include_answer": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer") or ""
                snippets = [
                    item.get("content") or item.get("snippet") or ""
                    for item in data.get("results", [])
                ]
                combined = " ".join([answer] + snippets).strip()
                if combined:
                    return _compact(combined, limit=400)
        except Exception:
            pass

    # Try SerpAPI as fallback
    serp_key = os.getenv("SERPAPI_API_KEY")
    if serp_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.get(
                    "https://serpapi.com/search",
                    params={
                        "api_key": serp_key,
                        "engine": "google",
                        "num": 5,
                        "q": query,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                organic = data.get("organic_results", [])
                snippets = [item.get("snippet", "") for item in organic]
                combined = " ".join(snippets).strip()
                if combined:
                    return _compact(combined, limit=400)
        except Exception:
            pass

    return None


def _llm_fallback(instruction: str, context: Optional[str] = None) -> str:
    """Use LLM to generate response when search APIs aren't available."""
    prompt = instruction.strip()
    if context:
        prompt += "\nContext:\n" + context.strip()
    response = llm.invoke([
        SystemMessage(content="You are an intelligence analyst providing factual, detailed analysis."),
        HumanMessage(content=prompt),
    ])
    return _compact(response.content, limit=400)


def _with_prefix(prefix: str, summary: str) -> str:
    """Add prefix to summary."""
    text = f"{prefix}: {summary}" if prefix else summary
    return _compact(text, limit=400)


# ========================================
# INTELLIGENCE TOOLS
# ========================================

# Cultural Intelligence Tools
@tool
def cultural_analysis(location: str) -> str:
    """Analyze cultural norms, social hierarchy, language nuances, and local customs."""
    query = f"{location} culture social hierarchy customs etiquette language nuances communication style dress code"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} Cultural Analysis", summary)

    instruction = f"Provide detailed cultural intelligence for {location}: language nuances, social hierarchy, customs, etiquette, communication styles, and attire norms."
    return _llm_fallback(instruction)


@tool
def local_sentiment(location: str) -> str:
    """Analyze current local sentiment, social tensions, and public mood."""
    query = f"{location} current local sentiment social tensions public opinion mood 2025"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} Local Sentiment", summary)

    instruction = f"Analyze current local sentiment and social dynamics in {location}. Include any social tensions or public concerns."
    return _llm_fallback(instruction)


# Economic Intelligence Tools
@tool
def economic_landscape(location: str) -> str:
    """Analyze GDP, business environment, supply chains, and economic trends."""
    query = f"{location} economy GDP business environment supply chains economic indicators 2025"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} Economic Landscape", summary)

    instruction = f"Provide economic intelligence for {location}: GDP trends, business environment, key industries, supply chains, and market conditions."
    return _llm_fallback(instruction)


@tool
def business_networks(location: str) -> str:
    """Identify key business networks, major companies, and commercial relationships."""
    query = f"{location} major companies business networks commercial relationships key industries"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} Business Networks", summary)

    instruction = f"Identify key business networks, major companies, and commercial ecosystems in {location}."
    return _llm_fallback(instruction)


@tool
def hnwi_profiles(location: str) -> str:
    """Identify high net-worth individuals and influential business leaders."""
    query = f"{location} wealthy individuals business leaders billionaires influential entrepreneurs"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} HNWI Profiles", summary)

    instruction = f"Identify prominent high net-worth individuals and influential business leaders in {location}."
    return _llm_fallback(instruction)


# Political Intelligence Tools
@tool
def political_structure(location: str) -> str:
    """Analyze government structure, key officials, and political system."""
    query = f"{location} government structure political system key officials leadership 2025"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} Political Structure", summary)

    instruction = f"Analyze the political structure of {location}: government type, key officials, political parties, and power dynamics."
    return _llm_fallback(instruction)


@tool
def political_tensions(location: str) -> str:
    """Identify political tensions, conflicts, and policy debates."""
    query = f"{location} political tensions conflicts policy debates current issues 2025"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} Political Tensions", summary)

    instruction = f"Identify current political tensions, conflicts, and major policy debates in {location}."
    return _llm_fallback(instruction)


# Security Intelligence Tools
@tool
def crime_analysis(location: str) -> str:
    """Analyze crime patterns, threat zones, and safety concerns."""
    query = f"{location} crime statistics safety threats dangerous areas security concerns 2025"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} Crime Analysis", summary)

    instruction = f"Analyze crime patterns, threat levels, and safety concerns in {location}. Include high-risk areas and security recommendations."
    return _llm_fallback(instruction)


@tool
def criminal_players(location: str) -> str:
    """Identify key criminal organizations and security threats."""
    query = f"{location} organized crime criminal organizations security threats gangs"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} Criminal Players", summary)

    instruction = f"Identify major criminal organizations, security threats, and key players in the criminal underworld of {location}."
    return _llm_fallback(instruction)


@tool
def security_infrastructure(location: str) -> str:
    """Analyze security infrastructure, police presence, and emergency services."""
    query = f"{location} police security infrastructure emergency services law enforcement"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} Security Infrastructure", summary)

    instruction = f"Analyze security infrastructure in {location}: police presence, emergency services, surveillance systems, and law enforcement capabilities."
    return _llm_fallback(instruction)


# Operational Intelligence Tools
@tool
def poi_mapping(location: str) -> str:
    """Map points of interest for operational awareness."""
    query = f"{location} important locations landmarks embassies hospitals police stations"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} POI Mapping", summary)

    instruction = f"Identify key points of interest in {location}: embassies, safe houses, hospitals, police stations, transportation hubs, and strategic locations."
    return _llm_fallback(instruction)


@tool
def support_resources(location: str) -> str:
    """Identify support resources, contacts, and emergency services."""
    query = f"{location} emergency contacts support resources embassy consulate hospitals"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} Support Resources", summary)

    instruction = f"List critical support resources in {location}: embassy contacts, medical facilities, safe accommodations, and emergency services."
    return _llm_fallback(instruction)


@tool
def logistics_intel(location: str) -> str:
    """Analyze logistics, transportation, and communication systems."""
    query = f"{location} transportation logistics communication systems infrastructure"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} Logistics", summary)

    instruction = f"Analyze logistics and infrastructure in {location}: transportation systems, communication networks, and operational considerations."
    return _llm_fallback(instruction)


@tool
def weather_conditions(location: str) -> str:
    """Get weather conditions and climate considerations."""
    query = f"{location} weather climate conditions temperature rainfall 2025"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} Weather", summary)

    instruction = f"Provide weather and climate information for {location}: current conditions, seasonal patterns, and operational considerations."
    return _llm_fallback(instruction)


# Events Intelligence Tools
@tool
def local_events(location: str) -> str:
    """Identify upcoming local events, gatherings, and public activities."""
    query = f"{location} upcoming events festivals gatherings 2025"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} Local Events", summary)

    instruction = f"Identify upcoming events, festivals, and public gatherings in {location}."
    return _llm_fallback(instruction)


@tool
def cultural_celebrations(location: str) -> str:
    """Identify cultural celebrations, holidays, and traditional events."""
    query = f"{location} cultural celebrations holidays traditional festivals"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} Cultural Celebrations", summary)

    instruction = f"List important cultural celebrations, holidays, and traditional events in {location}."
    return _llm_fallback(instruction)


@tool
def political_events(location: str) -> str:
    """Identify political events, elections, and government activities."""
    query = f"{location} political events elections government meetings rallies 2025"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{location} Political Events", summary)

    instruction = f"Identify upcoming political events, elections, and government activities in {location}."
    return _llm_fallback(instruction)


# ========================================
# STATE DEFINITION
# ========================================

class DossierState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    dossier_request: Dict[str, Any]
    cultural: Optional[str]
    economic: Optional[str]
    political: Optional[str]
    security: Optional[str]
    operational: Optional[str]
    events: Optional[str]
    executive_summary: Optional[str]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]
    poi_profiles: Annotated[List[Dict[str, Any]], operator.add]


# ========================================
# INTELLIGENCE AGENTS
# ========================================

def cultural_agent(state: DossierState) -> DossierState:
    """Cultural Intelligence Agent: Analyzes cultural norms, social dynamics, and local customs."""
    req = state["dossier_request"]
    location = req["location"]

    prompt_t = (
        "You are a Cultural Intelligence Analyst.\n"
        "Provide comprehensive cultural intelligence for {location}.\n"
        "Cover: language nuances, social hierarchy, customs, local sentiment, attire norms, communication styles.\n"
        "Use tools to gather current information, then synthesize into actionable intelligence."
    )
    vars_ = {"location": location}

    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [cultural_analysis, local_sentiment]
    agent = llm.bind_tools(tools)

    calls: List[Dict[str, Any]] = []

    with using_attributes(tags=["cultural", "intelligence"]):
        with using_metadata(agent_type="cultural", agent_node="cultural_agent", location=location):
            with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
                res = agent.invoke(messages)

    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "cultural", "tool": c["name"], "args": c.get("args", {})})

        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})

        messages.append(res)
        messages.extend(tr["messages"])

        synthesis_prompt = f"Synthesize the cultural intelligence for {location} into a comprehensive brief suitable for field operatives."
        messages.append(SystemMessage(content=synthesis_prompt))

        with using_prompt_template(template=synthesis_prompt, variables=vars_, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "cultural": out, "tool_calls": calls}


def economic_agent(state: DossierState) -> DossierState:
    """Economic Intelligence Agent: Analyzes economic landscape and business networks."""
    req = state["dossier_request"]
    location = req["location"]

    prompt_t = (
        "You are an Economic Intelligence Analyst.\n"
        "Provide comprehensive economic intelligence for {location}.\n"
        "Cover: GDP trends, supply chains, business networks, HNWI profiles, currency/markets.\n"
        "Use tools to gather current information, then synthesize into actionable intelligence."
    )
    vars_ = {"location": location}

    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [economic_landscape, business_networks, hnwi_profiles]
    agent = llm.bind_tools(tools)

    calls: List[Dict[str, Any]] = []

    with using_attributes(tags=["economic", "intelligence"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "economic")
                current_span.set_attribute("metadata.agent_node", "economic_agent")

        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)

    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "economic", "tool": c["name"], "args": c.get("args", {})})

        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})

        messages.append(res)
        messages.extend(tr["messages"])

        synthesis_prompt = f"Synthesize the economic intelligence for {location} into a comprehensive brief."
        messages.append(SystemMessage(content=synthesis_prompt))

        with using_prompt_template(template=synthesis_prompt, variables=vars_, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "economic": out, "tool_calls": calls}


def political_agent(state: DossierState) -> DossierState:
    """Political Intelligence Agent: Analyzes political structure and tensions."""
    req = state["dossier_request"]
    location = req["location"]

    prompt_t = (
        "You are a Political Intelligence Analyst.\n"
        "Provide comprehensive political intelligence for {location}.\n"
        "Cover: government structure, key officials, policy trends, political tensions.\n"
        "Use tools to gather current information, then synthesize into actionable intelligence."
    )
    vars_ = {"location": location}

    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [political_structure, political_tensions]
    agent = llm.bind_tools(tools)

    calls: List[Dict[str, Any]] = []

    with using_attributes(tags=["political", "intelligence"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "political")
                current_span.set_attribute("metadata.agent_node", "political_agent")

        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)

    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "political", "tool": c["name"], "args": c.get("args", {})})

        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})

        messages.append(res)
        messages.extend(tr["messages"])

        synthesis_prompt = f"Synthesize the political intelligence for {location} into a comprehensive brief."
        messages.append(SystemMessage(content=synthesis_prompt))

        with using_prompt_template(template=synthesis_prompt, variables=vars_, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "political": out, "tool_calls": calls}


def security_agent(state: DossierState) -> DossierState:
    """Security Intelligence Agent: Analyzes threats, crime patterns, and security infrastructure."""
    req = state["dossier_request"]
    location = req["location"]

    prompt_t = (
        "You are a Security Intelligence Analyst.\n"
        "Provide comprehensive security intelligence for {location}.\n"
        "Cover: crime patterns, key criminal players, threat zones, security infrastructure.\n"
        "Use tools to gather current information, then synthesize into actionable intelligence."
    )
    vars_ = {"location": location}

    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [crime_analysis, criminal_players, security_infrastructure]
    agent = llm.bind_tools(tools)

    calls: List[Dict[str, Any]] = []

    with using_attributes(tags=["security", "intelligence"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "security")
                current_span.set_attribute("metadata.agent_node", "security_agent")

        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)

    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "security", "tool": c["name"], "args": c.get("args", {})})

        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})

        messages.append(res)
        messages.extend(tr["messages"])

        synthesis_prompt = f"Synthesize the security intelligence for {location} into a comprehensive brief."
        messages.append(SystemMessage(content=synthesis_prompt))

        with using_prompt_template(template=synthesis_prompt, variables=vars_, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "security": out, "tool_calls": calls}


def operational_agent(state: DossierState) -> DossierState:
    """Operational Intelligence Agent: Analyzes logistics, POIs, and support resources."""
    req = state["dossier_request"]
    location = req["location"]

    prompt_t = (
        "You are an Operational Intelligence Analyst.\n"
        "Provide comprehensive operational intelligence for {location}.\n"
        "Cover: POI mapping, support resources, logistics, communication systems, weather.\n"
        "Use tools to gather current information, then synthesize into actionable intelligence."
    )
    vars_ = {"location": location}

    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [poi_mapping, support_resources, logistics_intel, weather_conditions]
    agent = llm.bind_tools(tools)

    calls: List[Dict[str, Any]] = []

    with using_attributes(tags=["operational", "intelligence"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "operational")
                current_span.set_attribute("metadata.agent_node", "operational_agent")

        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)

    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "operational", "tool": c["name"], "args": c.get("args", {})})

        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})

        messages.append(res)
        messages.extend(tr["messages"])

        synthesis_prompt = f"Synthesize the operational intelligence for {location} into a comprehensive brief."
        messages.append(SystemMessage(content=synthesis_prompt))

        with using_prompt_template(template=synthesis_prompt, variables=vars_, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "operational": out, "tool_calls": calls}


def events_agent(state: DossierState) -> DossierState:
    """Events Intelligence Agent: Tracks local events, celebrations, and gatherings."""
    req = state["dossier_request"]
    location = req["location"]

    prompt_t = (
        "You are an Events Intelligence Analyst.\n"
        "Provide comprehensive events intelligence for {location}.\n"
        "Cover: local events calendar, cultural celebrations, political gatherings.\n"
        "Use tools to gather current information, then synthesize into actionable intelligence."
    )
    vars_ = {"location": location}

    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [local_events, cultural_celebrations, political_events]
    agent = llm.bind_tools(tools)

    calls: List[Dict[str, Any]] = []

    with using_attributes(tags=["events", "intelligence"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "events")
                current_span.set_attribute("metadata.agent_node", "events_agent")

        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)

    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "events", "tool": c["name"], "args": c.get("args", {})})

        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})

        messages.append(res)
        messages.extend(tr["messages"])

        synthesis_prompt = f"Synthesize the events intelligence for {location} into a comprehensive brief."
        messages.append(SystemMessage(content=synthesis_prompt))

        with using_prompt_template(template=synthesis_prompt, variables=vars_, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "events": out, "tool_calls": calls}


def synthesis_agent(state: DossierState) -> DossierState:
    """Synthesis Agent: Compiles all intelligence into comprehensive dossier with executive summary."""
    req = state["dossier_request"]
    location = req["location"]
    mission_duration = req.get("mission_duration", "indefinite")
    mission_objectives = req.get("mission_objectives", "general intelligence gathering")

    prompt_t = (
        "You are the Chief Intelligence Officer.\n"
        "Compile all intelligence reports into a comprehensive dossier for {location}.\n"
        "Mission Duration: {mission_duration}\n"
        "Mission Objectives: {mission_objectives}\n\n"
        "Available Intelligence:\n"
        "Cultural: {cultural}\n"
        "Economic: {economic}\n"
        "Political: {political}\n"
        "Security: {security}\n"
        "Operational: {operational}\n"
        "Events: {events}\n\n"
        "Create an executive summary (2-3 paragraphs) that highlights:\n"
        "1. Key operational considerations\n"
        "2. Primary threats and opportunities\n"
        "3. Critical situational awareness points\n"
        "4. Strategic recommendations\n"
    )

    vars_ = {
        "location": location,
        "mission_duration": mission_duration,
        "mission_objectives": mission_objectives,
        "cultural": (state.get("cultural") or "N/A")[:500],
        "economic": (state.get("economic") or "N/A")[:500],
        "political": (state.get("political") or "N/A")[:500],
        "security": (state.get("security") or "N/A")[:500],
        "operational": (state.get("operational") or "N/A")[:500],
        "events": (state.get("events") or "N/A")[:500],
    }

    with using_attributes(tags=["synthesis", "executive"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "synthesis")
                current_span.set_attribute("metadata.agent_node", "synthesis_agent")

        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = llm.invoke([SystemMessage(content=prompt_t.format(**vars_))])

    return {
        "messages": [SystemMessage(content=res.content)],
        "executive_summary": res.content,
    }


# ========================================
# LANGGRAPH WORKFLOW
# ========================================

def build_dossier_graph():
    """Build the multi-agent intelligence dossier graph."""
    g = StateGraph(DossierState)

    # Add all agent nodes
    g.add_node("cultural_node", cultural_agent)
    g.add_node("economic_node", economic_agent)
    g.add_node("political_node", political_agent)
    g.add_node("security_node", security_agent)
    g.add_node("operational_node", operational_agent)
    g.add_node("events_node", events_agent)
    g.add_node("synthesis_node", synthesis_agent)

    # Parallel execution: All 6 intelligence agents run simultaneously
    g.add_edge(START, "cultural_node")
    g.add_edge(START, "economic_node")
    g.add_edge(START, "political_node")
    g.add_edge(START, "security_node")
    g.add_edge(START, "operational_node")
    g.add_edge(START, "events_node")

    # All agents feed into synthesis agent
    g.add_edge("cultural_node", "synthesis_node")
    g.add_edge("economic_node", "synthesis_node")
    g.add_edge("political_node", "synthesis_node")
    g.add_edge("security_node", "synthesis_node")
    g.add_edge("operational_node", "synthesis_node")
    g.add_edge("events_node", "synthesis_node")

    g.add_edge("synthesis_node", END)

    return g.compile()


# ========================================
# FASTAPI APPLICATION
# ========================================

app = FastAPI(title="Intelligence Dossier Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    """Serve the dossier frontend interface."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "frontend", "dossier.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "frontend/dossier.html not found"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "intelligence-dossier-agent"}


# Initialize Arize AX tracing at startup
if _TRACING:
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        if space_id and api_key:
            print(f"🔍 Initializing Arize AX observability for project: intelligence-dossier-agent")
            print(f"📊 Traces will be available at: https://app.arize.com/")

            # Register tracer provider with Arize
            tp = register(
                space_id=space_id,
                api_key=api_key,
                project_name="intelligence-dossier-agent",
                model_id="dossier-agent-v1",
            )

            # Auto-instrument LangChain for complete graph/agent/tool tracing
            LangChainInstrumentor().instrument(
                tracer_provider=tp,
                include_chains=True,
                include_agents=True,
                include_tools=True
            )

            # Auto-instrument LiteLLM for LLM call tracing
            LiteLLMInstrumentor().instrument(
                tracer_provider=tp,
                skip_dep_check=True
            )

            print("✅ Arize AX instrumentation complete - all agents will be traced")
        else:
            print("⚠️  Arize credentials not found. Set ARIZE_SPACE_ID and ARIZE_API_KEY to enable tracing")
    except Exception as e:
        print(f"⚠️  Failed to initialize Arize tracing: {e}")


@app.post("/generate-dossier", response_model=DossierResponse)
def generate_dossier(req: DossierRequest):
    """Generate comprehensive intelligence dossier for a location."""
    graph = build_dossier_graph()

    state = {
        "messages": [],
        "dossier_request": req.model_dump(),
        "tool_calls": [],
        "poi_profiles": [],
    }

    # Add session tracking and root span
    attrs_kwargs = {}
    if req.session_id:
        attrs_kwargs["session_id"] = req.session_id
    if req.user_id:
        attrs_kwargs["user_id"] = req.user_id

    # Create root span for the entire dossier generation workflow
    tracer = trace_api.get_tracer(__name__) if _TRACING else None

    if _TRACING and tracer:
        with tracer.start_as_current_span(
            "generate_dossier",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value if hasattr(OpenInferenceSpanKindValues.CHAIN, 'value') else "CHAIN",
                SpanAttributes.INPUT_VALUE: f"Generate intelligence dossier for {req.location}",
                "dossier.location": req.location,
                "dossier.mission_duration": req.mission_duration or "not_specified",
                "dossier.risk_tolerance": req.risk_tolerance,
            }
        ) as root_span:
            with using_attributes(**attrs_kwargs):
                out = graph.invoke(state)

            # Set output on root span
            root_span.set_attribute(SpanAttributes.OUTPUT_VALUE, out.get("executive_summary", "")[:500])
    else:
        with using_attributes(**attrs_kwargs):
            out = graph.invoke(state)

    return DossierResponse(
        executive_summary=out.get("executive_summary", ""),
        cultural_intel=out.get("cultural", ""),
        economic_intel=out.get("economic", ""),
        political_intel=out.get("political", ""),
        security_intel=out.get("security", ""),
        operational_intel=out.get("operational", ""),
        events_intel=out.get("events", ""),
        poi_profiles=out.get("poi_profiles", []),
        tool_calls=out.get("tool_calls", []),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
