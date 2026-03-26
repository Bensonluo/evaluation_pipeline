FROM apache/airflow:2.7.0-python3.11

COPY requirements-airflow.txt /tmp/requirements.txt
USER airflow
RUN pip install --no-cache-dir --timeout 300 -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt
