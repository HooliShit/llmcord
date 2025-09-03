FROM python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD sh -c "python keepalive.py & python llmcord.py"
