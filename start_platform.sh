#!/bin/bash

# Quantum-Aero F1 Prototype - Quick Start Script
# Starts all services with docker-compose

set -e

echo "🚀 Starting Quantum-Aero F1 Prototype Platform..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo -e "${BLUE}Step 1:${NC} Building and starting core services..."
docker-compose up -d --build

echo ""
echo -e "${BLUE}Step 2:${NC} Waiting for services to be healthy..."
sleep 10

# Check service health
check_service() {
    SERVICE=$1
    PORT=$2
    echo -n "Checking $SERVICE on port $PORT... "
    if curl -f http://localhost:$PORT/health > /dev/null 2>&1 || curl -f http://localhost:$PORT > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ (may take a few more seconds)${NC}"
        return 1
    fi
}

echo ""
check_service "MongoDB" 27017
check_service "Redis" 6379
check_service "Physics Engine (VLM)" 8001
check_service "Backend API" 3001
check_service "Frontend" 3000
check_service "NATS" 8222

echo ""
echo -e "${GREEN}✅ Platform is starting up!${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}📊 Access Points:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "  🌐 Frontend:          ${GREEN}http://localhost:3000${NC}"
echo -e "  🔧 Backend API:       ${GREEN}http://localhost:3001${NC}"
echo -e "  ⚡ VLM Solver:        ${GREEN}http://localhost:8001${NC}"
echo -e "  📡 NATS Monitoring:   ${GREEN}http://localhost:8222${NC}"
echo -e "  🗄️  MongoDB:          ${GREEN}mongodb://localhost:27017${NC}"
echo -e "  🔴 Redis:             ${GREEN}redis://localhost:6379${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}🎯 Quick Actions:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  View logs:           docker-compose logs -f"
echo "  Stop services:       docker-compose down"
echo "  Restart:             docker-compose restart"
echo "  Start GenAI agents:  docker-compose -f docker-compose.agents.yml up -d"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${YELLOW}Note:${NC} First build may take 5-10 minutes."
echo -e "${YELLOW}Tip:${NC}  Wait ~30 seconds for all services to fully initialize."
echo ""
