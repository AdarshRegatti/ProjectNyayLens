FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed for FAISS / build steps
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend application source code
COPY . .

# Expose the default port for Hugging Face Spaces
EXPOSE 7860

# Run Uvicorn pointing to your FastAPI entry point on port 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
