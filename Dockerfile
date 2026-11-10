FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data
WORKDIR /app
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt && useradd --uid 10001 --create-home app
COPY psb_backend.py web_api.py solution.py ./
COPY ML1/retrieval ./ML1/retrieval
COPY ML2/solution.py ./ML2/solution.py
COPY ML2/llmguard ./ML2/llmguard
COPY ML2/prompts ./ML2/prompts
USER 10001:10001
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "web_api:app", "--host", "0.0.0.0", "--port", "8000"]
