FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# CPU-only torch first -- see setup_venv.sh comment for why: plain `pip
# install torch` pulls 3+ GB of unneeded CUDA wheels on Linux.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the ChromaDB index at image build time so /ask works immediately.
# (Requires network access to download the embedding model on first build.)
RUN python ingest.py

ENV MOCK_LLM=1

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
