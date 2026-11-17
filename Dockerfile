# CPU image for the inference API. Weights are mounted at runtime, not baked in.
#   docker build -t visionops-api .
#   docker run -p 8000:8000 -v %cd%/runs:/app/runs visionops-api
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY visionops ./visionops
COPY api ./api
COPY configs ./configs
ENV VISIONOPS_MLFLOW=0
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
