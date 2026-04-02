FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/project
CMD ["uvicorn", "source.webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
