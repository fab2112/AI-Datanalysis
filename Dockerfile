# base image
FROM python:3.11-slim

# Python variable
ENV PYTHONUNBUFFERED=1

# Make work dir
RUN mkdir -p /home/appuser

# Set work dir
WORKDIR /home/appuser

# Set user and usergroup
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

# Add virtualenv to PATH
ENV PATH="/home/appuser/venv/bin:${PATH}"

# Copy project
COPY /.streamlit  /home/appuser/.streamlit
COPY /css  /home/appuser/css
COPY /ragdata  /home/appuser/ragdata
COPY app.py /home/appuser
COPY agent.py /home/appuser
COPY prompt.py /home/appuser
COPY styles.py /home/appuser
COPY poetry.lock /home/appuser
COPY pyproject.toml /home/appuser

# Install poetry, dependences and activate venv
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir poetry && \
    poetry install --only main --no-interaction --no-ansi 

RUN chown -R appuser:appgroup /home/appuser

# Set user
USER appuser

# Start application
ENTRYPOINT ["streamlit", "run", "app.py"]
