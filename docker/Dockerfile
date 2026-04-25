# Multi-stage build for FairSight

# Stage 1: Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# Stage 2: Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 fairsight

# Copy installed packages from builder
COPY --from=builder /root/.local /home/fairsight/.local

# Copy application code
COPY . .

# Set ownership
RUN chown -R fairsight:fairsight /app

# Switch to non-root user
USER fairsight

# Set Python path
ENV PATH=/home/fairsight/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=2)" || exit 1

# Expose port
EXPOSE 8000

# Run FastAPI application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
