#!/bin/bash

# Alternative deployment without Cloud Build
# Requires Docker and gcloud CLI

# Build the Docker image locally
docker build -t fairsight-backend .

# Tag for Container Registry
docker tag fairsight-backend gcr.io/plasma-line-471215-r9/fairsight-backend:latest

# Push to Container Registry (requires billing)
docker push gcr.io/plasma-line-471215-r9/fairsight-backend:latest

# Deploy to Cloud Run
gcloud run deploy fairsight-backend \
  --image=gcr.io/plasma-line-471215-r9/fairsight-backend:latest \
  --region=us-central1 \
  --platform=managed \
  --allow-unauthenticated \
  --port=8001 \
  --memory=512Mi \
  --cpu=1
