FROM python:3.11-slim

WORKDIR /app

# install dependencies first so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY . .

# make sample_outputs writable at runtime
RUN mkdir -p sample_outputs

# default: run the pipeline
CMD ["python", "run_pipeline.py"]
