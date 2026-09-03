#!/bin/bash
set -e

# Frontend Setup Script for MARS Mock Pipeline
# Usage: cd frontend && bash setup.sh

echo "🚀 Setting up MARS frontend..."

# Check if we're in the frontend directory
if [ ! -f "package.json" ]; then
  echo "❌ Error: package.json not found. Please run this script from the frontend/ directory."
  exit 1
fi

echo "📦 Installing dependencies..."
npm install

echo "🧪 Running tests..."
npm test

echo "✅ Frontend setup complete!"
echo ""
echo "Next steps:"
echo "  - Run tests: npm test"
echo "  - Start dev server: npm run dev (requires App.tsx from Task 12)"
echo "  - Build for production: npm run build (requires App.tsx from Task 12)"
