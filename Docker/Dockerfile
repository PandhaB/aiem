FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY engine /app/engine
COPY core /app/core
COPY tests /app/tests
COPY pytest.ini /app/pytest.ini
COPY Docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONPATH=/app
ENV AITEM_PROJECTS_DIR=/data/projects
ENV AITEM_DATASETS_DIR=/data/datasets
ENV AITEM_WEIGHTS_DIR=/data/weights

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
