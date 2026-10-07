#!/bin/bash
set -e

# MARS Mock Pipeline - Complete Setup Script
# Usage: bash script.sh [backend|frontend|all|test]

COMMAND="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 MARS Mock Pipeline Setup"
echo "=============================="
echo ""

setup_backend() {
  echo "📦 Setting up backend..."
  if [ ! -f "$SCRIPT_DIR/backend/pyproject.toml" ]; then
    echo "❌ Error: backend/pyproject.toml not found"
    exit 1
  fi

  echo "Installing Python dependencies..."
  # backend/venv has no pip available (python3-venv isn't installed on this
  # system), so install into the system interpreter's user site instead —
  # this is how the project is already set up (see AGENT.md).
  python3 -m pip install -q -e "$SCRIPT_DIR/backend[dev]" --break-system-packages

  echo "✅ Backend setup complete"
  echo "   Run tests: cd backend && python3 -m pytest"
  echo "   Run API:   cd backend && python3 -m uvicorn mars.api:app --reload"
}

setup_frontend() {
  echo "📦 Setting up frontend..."
  (cd "$SCRIPT_DIR/frontend" && npm install)
  echo "✅ Frontend setup complete"
}

run_tests() {
  echo "🧪 Running backend tests..."
  (cd "$SCRIPT_DIR/backend" && python3 -m pytest)

  echo ""
  echo "🧪 Running frontend tests..."
  (cd "$SCRIPT_DIR/frontend" && npm test)
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
  test)
    run_tests
    ;;
  *)
    echo "Usage: bash script.sh [backend|frontend|all|test]"
    echo ""
    echo "Options:"
    echo "  backend  - Setup Python backend only (creates backend/venv)"
    echo "  frontend - Setup frontend (Node/npm) only"
    echo "  all      - Setup both backend and frontend (default)"
    echo "  test     - Run backend and frontend test suites"
    exit 1
    ;;
esac

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start developing:"
echo "  Backend:  cd backend && python3 -m uvicorn mars.api:app --reload"
echo "  Frontend: cd frontend && npm run dev"
