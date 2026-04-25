# Google Cloud Deployment Guide for FairSight Core

This guide helps you deploy FairSight Core to Google Cloud and integrate Google Gemini AI for the Google Solution Challenge.

## Prerequisites

1. **Google Cloud Account** (Free tier available)
   - Create account at: https://cloud.google.com/free
   - $300 free credit for new users
   - Free tier includes: Cloud Run, Cloud Storage, Cloud Build

2. **Google AI API Key** (Free tier)
   - Get API key at: https://makersuite.google.com/app/apikey
   - Free tier: 15 requests/minute for Gemini Pro

3. **Google Cloud SDK**
   - Install: https://cloud.google.com/sdk/docs/install
   - Authenticate: `gcloud auth login`

## Step 1: Set Up Google Cloud Project

```bash
# Create a new project
gcloud projects create fairsight-solution --name="FairSight Solution"

# Set as default project
gcloud config set project fairsight-solution

# Enable required APIs
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com
```

## Step 2: Configure Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Google API key
# GOOGLE_API_KEY=your_actual_api_key_here
```

## Step 3: Deploy Backend to Cloud Run

```bash
# Build and deploy using Cloud Build
gcloud builds submit --config cloudbuild.yaml .

# Get the service URL
gcloud run services describe fairsight-backend --region=us-central1 --format='value(status.url)'
```

**Free Tier Limits:**
- 2 million requests/month
- 200,000 vCPU-seconds/month
- 1 GB memory per instance

## Step 4: Deploy Frontend to Cloud Storage (Static Site)

```bash
# Build the frontend
cd frontend
npm install
npm run build

# Create a Cloud Storage bucket
gsutil mb -p fairsight-solution gs://fairsight-frontend

# Upload the built files
gsutil -m cp -r dist/* gs://fairsight-frontend/

# Make the bucket public
gsutil iam ch allUsers:objectViewer gs://fairsight-frontend

# Enable website hosting
gsutil web set -m index.html -e 404.html gs://fairsight-frontend
```

**Free Tier Limits:**
- 5 GB storage
- 1 GB egress/month

## Step 5: Update Frontend API URL

Edit `frontend/.env`:
```
VITE_API_URL=https://your-cloud-run-url.cloudfunctions.net
```

## Step 6: Verify Google AI Integration

The backend automatically uses Google Gemini API when:
- `GOOGLE_API_KEY` is set in environment variables
- AI insights are generated for each bias metric

**Features powered by Gemini AI:**
- Enhanced bias explanations
- Context-aware recommendations
- Actionable insights based on severity

## Step 7: Test the Deployment

```bash
# Test backend health
curl https://your-cloud-run-url/health

# Test frontend
open https://fairsight-frontend.storage.googleapis.com
```

## Google Solution Challenge Checklist

- [x] Deployed on Google Cloud (Cloud Run + Cloud Storage)
- [x] Using Google AI service (Gemini API - Free tier)
- [x] Free tier compliant (no costs during challenge)
- [x] Scalable architecture
- [x] Environment variables configured

## Cost Summary (Free Tier)

| Service | Free Tier Limit | Usage |
|---------|----------------|-------|
| Cloud Run | 2M requests/month | ~1000 requests/day |
| Cloud Storage | 5 GB storage | ~50 MB |
| Cloud Build | 120 minutes/day | ~10 minutes/month |
| Gemini API | 15 requests/min | ~5 requests/analysis |

**Total Monthly Cost: $0** (within free tier)

## Troubleshooting

**Build fails:**
```bash
# Check build logs
gcloud builds list --limit=1
gcloud builds log [BUILD_ID]
```

**API errors:**
- Verify GOOGLE_API_KEY is set correctly
- Check AI Platform API is enabled
- Verify API key has necessary permissions

**Deployment issues:**
```bash
# Check service status
gcloud run services describe fairsight-backend --region=us-central1

# View logs
gcloud logging read "resource.type=cloud_run_revision" --limit=50
```

## Security Notes

1. Never commit `.env` file to version control
2. Use Secret Manager for production API keys
3. Enable IAM authentication for Cloud Run in production
4. Set up Cloud Armor for DDoS protection

## Additional Resources

- Google Cloud Free Tier: https://cloud.google.com/free
- Gemini API Documentation: https://ai.google.dev/docs
- Cloud Run Documentation: https://cloud.google.com/run/docs
- Solution Challenge Guide: https://developers.google.com/solution-challenge
