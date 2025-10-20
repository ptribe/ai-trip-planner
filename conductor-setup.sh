#!/bin/bash
set -e  # Exit on error

echo "🔧 Setting up AI Trip Planner workspace..."

# Check Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "Please install uv: https://github.com/astral-sh/uv"
    echo "Quick install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Copy .env file from repository root
echo "📋 Copying environment file..."
if [ -f "$CONDUCTOR_ROOT_PATH/backend/.env" ]; then
    cp "$CONDUCTOR_ROOT_PATH/backend/.env" backend/.env
    echo "✅ Environment file copied"
else
    echo "⚠️  Warning: No .env file found at $CONDUCTOR_ROOT_PATH/backend/.env"
    echo "Creating template .env file - you'll need to add your API keys"
    cat > backend/.env << 'EOF'
# Required: Choose one LLM provider
OPENAI_API_KEY=your_openai_key_here
# OPENROUTER_API_KEY=your_openrouter_key_here

# Optional: Enable RAG for local experiences
# ENABLE_RAG=1

# Optional: Enable real-time web search
# TAVILY_API_KEY=your_tavily_key_here
# SERPAPI_API_KEY=your_serpapi_key_here

# Optional: Enable Arize tracing/observability
# ARIZE_SPACE_ID=your_space_id
# ARIZE_API_KEY=your_arize_key
EOF
    echo "⚠️  Please edit backend/.env and add your API keys before running"
fi

# Activate venv and install dependencies
echo "📦 Installing Python dependencies with uv..."
source venv/bin/activate
cd backend
uv pip install -r requirements.txt
cd ..

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Ensure backend/.env has your API keys (at minimum OPENAI_API_KEY or OPENROUTER_API_KEY)"
echo "2. Click 'Run' in Conductor to start the server"
echo "3. Access the app at http://localhost:8000"
