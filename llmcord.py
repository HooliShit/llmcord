# llmcord.py

import os
import discord
import asyncio
import datetime

# Variable globale pour suivre le dernier moment où une tâche a été exécutée
last_task_time = None
response_msgs = []

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


async def handle_message(message: discord.Message):
    global last_task_time, response_msgs   # ✅ utilisation de global
    # On met à jour l'état global
    last_task_time = datetime.datetime.utcnow()
    response_msgs.append(message.content)

    # Exemple de réponse
    await message.channel.send(f"Message reçu: {message.content}")


@client.event
async def on_ready():
    print(f"Connecté en tant que {client.user}")


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return
    await handle_message(message)


async def background_task():
    global last_task_time   # ✅ important si tu veux le modifier ici aussi
    await client.wait_until_ready()
    while not client.is_closed():
        if last_task_time is not None:
            now = datetime.datetime.utcnow()
            delta = (now - last_task_time).total_seconds()
            if delta > 60:  # plus d'une minute sans activité
                channel = discord.utils.get(client.get_all_channels(), name="general")
                if channel:
                    await channel.send("Pas d’activité depuis plus d’une minute...")
                last_task_time = None
        await asyncio.sleep(10)


if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_TOKEN")
    if not TOKEN:
        raise RuntimeError("⚠️ Variable d'environnement DISCORD_TOKEN manquante")
    client.loop.create_task(background_task())
    client.run(TOKEN)
