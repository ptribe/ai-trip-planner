#!/bin/bash

# Trip Planner Startup Script

echo "🚀 Starting AI Trip Planner..."

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the trip_planner directory"
    exit 1
fi

# Start backend in background
echo "📡 Starting backend server..."
cd backend
python main.py &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🎨 Starting frontend..."
cd ../frontend
npm start &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

echo ""
echo "✅ Services started successfully!"
echo "📡 Backend API: http://localhost:8000"
echo "🎨 Frontend: http://localhost:3000"
echo "📊 Arize Traces: https://app.arize.com/"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup when script is interrupted
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Services stopped."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for services
wait
