#!/bin/bash
set -e

# MARS Mock Pipeline - Complete Setup Script
# Usage: bash script.sh [backend|frontend|all]

COMMAND="${1:-all}"

echo "🚀 MARS Mock Pipeline Setup"
echo "=============================="
echo ""

setup_backend() {
  echo "📦 Setting up backend..."
  if [ ! -f "backend/requirements.txt" ]; then
    echo "❌ Error: backend/requirements.txt not found"
    exit 1
  fi

  echo "Installing Python dependencies..."
  pip install -r backend/requirements.txt

  echo "✅ Backend setup complete"
  echo "   Run: cd backend && python -m pytest"
  echo "   Or:  python -m uvicorn mars.api:app --reload"
}

setup_frontend() {
  echo "📦 Setting up frontend..."
  cd frontend
  bash setup.sh
  cd ..
}

case "$COMMAND" in
  backend)
    setup_backend
    ;;
  frontend)
    setup_frontend
    ;;
  all)
    setup_backend
    echo ""
    setup_frontend
    ;;
  *)
    echo "Usage: bash script.sh [backend|frontend|all]"
    echo ""
    echo "Options:"
    echo "  backend  - Setup Python backend only"
    echo "  frontend - Setup frontend (Node/npm) only"
    echo "  all      - Setup both backend and frontend (default)"
    exit 1
    ;;
esac

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start developing:"
echo "  Backend:  cd backend && python -m uvicorn mars.api:app --reload"
echo "  Frontend: cd frontend && npm run dev"
