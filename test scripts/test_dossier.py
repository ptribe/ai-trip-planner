"""
Test script for Intelligence Dossier Agent API
"""

import httpx
import json
from datetime import datetime


def test_dossier_generation(base_url: str = "http://localhost:8001"):
    """Test the dossier generation endpoint."""

    print("=" * 80)
    print("INTELLIGENCE DOSSIER AGENT - TEST SCRIPT")
    print("=" * 80)
    print(f"Base URL: {base_url}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Test payload
    payload = {
        "location": "Moscow, Russia",
        "mission_duration": "6 months",
        "mission_objectives": "Establish local network and gather economic intelligence",
        "risk_tolerance": "medium",
        "focus_areas": ["economic", "political", "security"],
        "session_id": "test-session-001",
        "user_id": "test-analyst"
    }

    print("REQUEST PAYLOAD:")
    print(json.dumps(payload, indent=2))
    print()

    # Make request
    print("Sending request to /generate-dossier...")
    print("(This may take 30-90 seconds due to parallel agent execution)")
    print()

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{base_url}/generate-dossier",
                json=payload,
            )
            response.raise_for_status()

            data = response.json()

            print("=" * 80)
            print("DOSSIER GENERATION SUCCESSFUL")
            print("=" * 80)
            print()

            # Print Executive Summary
            print("EXECUTIVE SUMMARY")
            print("-" * 80)
            print(data.get("executive_summary", "N/A"))
            print()

            # Print each intelligence domain
            domains = [
                ("CULTURAL INTELLIGENCE", "cultural_intel"),
                ("ECONOMIC INTELLIGENCE", "economic_intel"),
                ("POLITICAL INTELLIGENCE", "political_intel"),
                ("SECURITY INTELLIGENCE", "security_intel"),
                ("OPERATIONAL INTELLIGENCE", "operational_intel"),
                ("EVENTS INTELLIGENCE", "events_intel"),
            ]

            for title, key in domains:
                print(title)
                print("-" * 80)
                print(data.get(key, "N/A"))
                print()

            # Print tool calls summary
            tool_calls = data.get("tool_calls", [])
            if tool_calls:
                print("TOOL CALLS SUMMARY")
                print("-" * 80)
                print(f"Total tool calls: {len(tool_calls)}")
                print()

                # Group by agent
                by_agent = {}
                for call in tool_calls:
                    agent = call.get("agent", "unknown")
                    if agent not in by_agent:
                        by_agent[agent] = []
                    by_agent[agent].append(call.get("tool", "unknown"))

                for agent, tools in by_agent.items():
                    print(f"  {agent.upper()} Agent: {', '.join(tools)}")
                print()

            # Print POI profiles if any
            poi_profiles = data.get("poi_profiles", [])
            if poi_profiles:
                print("PERSON OF INTEREST PROFILES")
                print("-" * 80)
                for idx, profile in enumerate(poi_profiles, 1):
                    print(f"{idx}. {profile.get('name', 'Unknown')}")
                    print(f"   Position: {profile.get('position', 'N/A')}")
                    print(f"   Threat Level: {profile.get('threat_level', 'N/A')}")
                    print()

            print("=" * 80)
            print("TEST COMPLETED SUCCESSFULLY")
            print("=" * 80)

    except httpx.HTTPStatusError as e:
        print(f"HTTP ERROR: {e.response.status_code}")
        print(f"Response: {e.response.text}")
    except httpx.RequestError as e:
        print(f"REQUEST ERROR: {e}")
    except Exception as e:
        print(f"ERROR: {e}")


def test_health_check(base_url: str = "http://localhost:8001"):
    """Test the health endpoint."""
    print("Testing health endpoint...")
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/health")
            response.raise_for_status()
            data = response.json()
            print(f"✓ Health check passed: {data}")
            return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"

    # Run health check first
    if not test_health_check(base_url):
        print()
        print("Server is not running. Start it with:")
        print("  cd backend && python dossier_main.py")
        sys.exit(1)

    print()

    # Run dossier generation test
    test_dossier_generation(base_url)
