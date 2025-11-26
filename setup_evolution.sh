#!/bin/bash

# Evolution Components Setup Script
# Automated installation and configuration for Phase 1 components

set -e  # Exit on error

echo "=================================================="
echo "  Quantum-Aero F1 Evolution Setup"
echo "  Phase 1: Advanced AI Surrogates"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

if ! python3 -c 'import sys; assert sys.version_info >= (3, 9)' 2>/dev/null; then
    echo -e "${RED}✗ Python 3.9+ required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Check Node.js version
echo "Checking Node.js version..."
node_version=$(node --version 2>&1)
echo "  Node.js version: $node_version"

if ! node -e 'process.exit(parseInt(process.version.slice(1)) >= 18 ? 0 : 1)' 2>/dev/null; then
    echo -e "${RED}✗ Node.js 18+ required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js version OK${NC}"
echo ""

# Create virtual environment
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
echo "  This may take several minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Install PyTorch Geometric
echo "Installing PyTorch Geometric..."
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
echo -e "${GREEN}✓ PyTorch Geometric installed${NC}"
echo ""

# Install ML dependencies
echo "Installing ML dependencies..."
pip install \
    transformers>=4.36.0 \
    diffusers>=0.25.0 \
    stable-baselines3>=2.2.0 \
    tensorboard>=2.15.0 \
    h5py>=3.10.0 \
    trimesh>=4.0.0
echo -e "${GREEN}✓ ML dependencies installed${NC}"
echo ""

# Install Quantum dependencies
echo "Installing Quantum dependencies..."
pip install \
    qiskit>=1.0.0 \
    pennylane>=0.35.0 \
    dwave-ocean-sdk>=6.7.0
echo -e "${GREEN}✓ Quantum dependencies installed${NC}"
echo ""

# Install API dependencies
echo "Installing API dependencies..."
pip install \
    fastapi>=0.109.0 \
    uvicorn[standard]>=0.27.0 \
    pydantic>=2.5.0 \
    python-multipart>=0.0.6
echo -e "${GREEN}✓ API dependencies installed${NC}"
echo ""

# Install general dependencies
echo "Installing general dependencies..."
pip install \
    numpy>=1.24.0 \
    scipy>=1.11.0 \
    matplotlib>=3.8.0 \
    pandas>=2.1.0 \
    scikit-learn>=1.3.0 \
    tqdm>=4.66.0
echo -e "${GREEN}✓ General dependencies installed${NC}"
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/cfd_dataset/{train,val}
mkdir -p checkpoints/{aerotransformer,gnn_rans,vqe}
mkdir -p runs/{aerotransformer,gnn_rans}
mkdir -p logs
echo -e "${GREEN}✓ Directory structure created${NC}"
echo ""

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
    echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Frontend dependencies already installed${NC}"
fi
cd ..
echo ""

# Create .env file from template
echo "Creating .env file..."
if [ ! -f "agents/.env" ]; then
    cp agents/.env.template agents/.env
    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}⚠ Please edit agents/.env and add your API keys${NC}"
else
    echo -e "${YELLOW}⚠ .env file already exists${NC}"
fi
echo ""

# Test imports
echo "Testing Python imports..."
python3 << EOF
try:
    import torch
    import torch_geometric
    import transformers
    import qiskit
    import fastapi
    print("✓ All imports successful")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python environment verified${NC}"
else
    echo -e "${RED}✗ Python environment verification failed${NC}"
    exit 1
fi
echo ""

# Print summary
echo "=================================================="
echo "  Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Configure API keys in agents/.env"
echo ""
echo "  3. Start AeroTransformer API:"
echo "     python -m ml_service.models.aero_transformer.api"
echo ""
echo "  4. Start GNN-RANS API:"
echo "     python -m ml_service.models.gnn_rans.api"
echo ""
echo "  5. Start frontend:"
echo "     cd frontend && npm start"
echo ""
echo "  6. Access dashboards:"
echo "     http://localhost:3000"
echo ""
echo "=================================================="
echo ""

# Save installed packages
echo "Saving installed packages..."
pip freeze > requirements-evolution.txt
echo -e "${GREEN}✓ Requirements saved to requirements-evolution.txt${NC}"
echo ""

echo -e "${GREEN}Setup complete! 🎉${NC}"
