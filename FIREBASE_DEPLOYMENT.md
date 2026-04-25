# Firebase Deployment Alternative

If Google Cloud billing is an issue, you can use Firebase Hosting which is completely free and doesn't require billing setup.

## Prerequisites

```bash
# Install Firebase CLI
npm install -g firebase-tools

# Login to Firebase
firebase login
```

## Deploy Frontend to Firebase

```bash
cd frontend
npm install
npm run build

# Initialize Firebase in the project root
cd ..
firebase init hosting

# Select:
# - Create a new project
# - Public directory: frontend/dist
# - Configure as single-page app: Yes
# - Set up automatic builds: No

# Deploy
firebase deploy
```

## Backend Deployment

For the backend, you have options:
1. Use Google Cloud Run (requires billing)
2. Use Render.com (free tier)
3. Use Railway.app (free tier)
4. Use local development for demo

## Render.com Deployment (Free Tier)

```bash
# Create account at render.com
# Connect GitHub repository
# Deploy as Web Service
# Runtime: Python
# Build command: pip install -r requirements.txt
# Start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
# Add environment variable: GOOGLE_API_KEY
```

This approach meets Google Solution Challenge requirements by using Google Firebase (a Google service) for hosting.
