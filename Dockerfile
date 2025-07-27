FROM --platform=linux/amd64 python:3.11

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the processing script
COPY process_pdfs.py .

# Predownload the SentenceTransformer model to make it available offline
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('intfloat/e5-small'); model.save('/app/model')"

# Copy input and output directories
COPY input/ ./input/
# Run the script
CMD ["python", "process_pdfs.py"] 