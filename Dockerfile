FROM python:3.11-slim

# Dossier de travail DANS le conteneur
WORKDIR /app

# Dépendances (cache-friendly)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copie TOUT le code à /app (inclut llmcord.py, keepalive.py, config.yaml)
COPY . /app

# Render Web Service : doit écouter $PORT -> on expose un mini serveur HTTP
ENV PORT=10000
EXPOSE 10000

# Démarre le mini HTTP (healthcheck) + le bot
CMD sh -c "python keepalive.py & python llmcord.py"
