import os
import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_bot_token(cfg: dict[str, Any]) -> str:
    token_env = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    if token_env and not token_env.startswith("${"):
        return token_env
    token_cfg = (cfg.get("bot_token") or "").strip()
    if token_cfg:
        return token_cfg
    raise RuntimeError("Aucun token trouvé : définis DISCORD_BOT_TOKEN dans l'environnement ou bot_token dans config.yaml")


def resolve_api_key(provider_name: str, provider_cfg: dict[str, Any]) -> str:
    raw = (provider_cfg.get("api_key") or "").strip()

    if raw.startswith("${") and raw.endswith("}"):
        env_var = raw[2:-1].strip()
        val = os.environ.get(env_var, "").strip()
        if val:
            return val
        raise RuntimeError(f"Missing environment variable: {env_var}")

    fallback_env_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "x-ai": "XAI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
    }
    env_var = fallback_env_map.get(provider_name.lower())
    if env_var:
        val = os.environ.get(env_var, "").strip()
        if val:
            return val

    if raw and not raw.startswith("${"):
        return raw

    raise RuntimeError(f"No API key found for provider '{provider_name}'. Check your environment variables.")


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes: dict[int, "MsgNode"] = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True

status = (config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128]
activity = discord.CustomActivity(name=status)
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        cfg = await asyncio.to_thread(get_config)
        admins = (cfg.get("permissions", {}).get("users", {}) or {}).get("admin_ids", []) or []
        if interaction.user.id in admins:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"○ {m}", value=m) for m in config["models"] if m != curr_model and curr_str.lower() in m.lower()][:24]
    if curr_str.lower() in curr_model.lower():
        choices += [Choice(name=f"◉ {curr_model} (current)", value=curr_model)]

    return choices


@discord_bot.event
async def on_ready() -> None:
    client_id = os.environ.get("DISCORD_CLIENT_ID") or config.get("client_id")
    if client_id:
        logging.info(
            f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n"
        )
    await discord_bot.tree.sync()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = {r.id for r in getattr(new_msg.author, "roles", ())}
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    cfg = await asyncio.to_thread(get_config)

    # --- permissions robustes (avec valeurs par défaut) ---
    permissions = cfg.get("permissions", {}) or {}
    users_perm = permissions.get("users", {}) or {}
    roles_perm = permissions.get("roles", {}) or {}
    chans_perm = permissions.get("channels", {}) or {}

    admin_ids = users_perm.get("admin_ids", []) or []
    allowed_user_ids = users_perm.get("allowed_ids", []) or []
    blocked_user_ids = users_perm.get("blocked_ids", []) or []

    allowed_role_ids = roles_perm.get("allowed_ids", []) or []
    blocked_role_ids = roles_perm.get("blocked_ids", []) or []

    allowed_channel_ids = chans_perm.get("allowed_ids", []) or []
    blocked_channel_ids = chans_perm.get("blocked_ids", []) or []

    allow_dms = cfg.get("allow_dms", True)

    user_is_admin = new_msg.author.id in admin_ids

    allow_all_users = (not allowed_user_ids) if is_dm else (not allowed_user_ids and not allowed_role_ids)
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(i in allowed_role_ids for i in role_ids)
    is_bad_user = (not is_good_user) or (new_msg.author.id in blocked_user_ids) or any(i in blocked_role_ids for i in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = (user_is_admin or allow_dms) if is_dm else (allow_all_channels or any(i in allowed_channel_ids for i in channel_ids))
    is_bad_channel = (not is_good_channel) or any(i in blocked_channel_ids for i in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = cfg["providers"][provider]

    base_url = provider_config["base_url"].strip()
    api_key = resolve_api_key(provider, provider_config)

    # Client OpenAI-compatible avec timeouts raisonnables
    openai_client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=60.0,  # secondes
        max_retries=2,
        http_client=httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    )

    model_parameters = cfg["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers", None)
    extra_query = provider_config.get("extra_query", None)

    # ---- normalisation des paramètres pour l'API OpenAI-compatible ----
    extra_body = {}
    extra_body.update(provider_config.get("extra_body", {}) or {})
    extra_body.update(model_parameters or {})
    if "max_output_tokens" in extra_body and "max_tokens" not in extra_body:
        extra_body["max_tokens"] = extra_body.pop("max_output_tokens")
    if not extra_body:
        extra_body = None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = cfg.get("max_text", 100000)
    max_images = cfg.get("max_images", 5) if accept_images else 0
    max_messages = cfg.get("max_messages", 25)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg is not None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text is None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_atts = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]
                att_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_atts])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_atts, att_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_atts, att_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"
                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_atts)

                try:
                    if (
                        curr_msg.reference is None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference is None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := (curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None)):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id is not None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text or "") > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg is not None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(
        f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}"
    )

    # Ajout du prompt système
    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()
        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        if accept_usernames:
            system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."
        messages.append(dict(role="system", content=system_prompt))

    # Prépare la réponse
    response_msgs: list[discord.Message] = []
    response_contents: list[str] = []

    # --- IMPORTANT : pas de streaming sur OpenRouter (workaround blocages) ---
    use_stream = (provider.lower() != "openrouter")

    openai_kwargs = dict(
        model=model,
        messages=messages[::-1],
        stream=use_stream,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body
    )

    async def reply_helper_embed(text: str, final: bool) -> None:
        nonlocal response_msgs, last_task_time
        embed = discord.Embed(description=text, color=(EMBED_COLOR_COMPLETE if final else EMBED_COLOR_INCOMPLETE))
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        if not response_msgs:
            msg = await reply_target.reply(embed=embed, silent=True)
            response_msgs.append(msg)
        else:
            await response_msgs[-1].edit(embed=embed)
        last_task_time = datetime.now().timestamp()

    try:
        async with new_msg.channel.typing():
            if use_stream:
                # Streaming (Groq, autres)
                max_message_length = 4096 - len(STREAMING_INDICATOR)
                await reply_helper_embed(STREAMING_INDICATOR, final=False)

                curr_content = ""
                got_any_chunk = False
                stream = await openai_client.chat.completions.create(**openai_kwargs)

                async for chunk in stream:
                    choice = chunk.choices[0] if chunk.choices else None
                    if not choice:
                        continue
                    delta = choice.delta.content or ""
                    if not delta:
                        continue
                    got_any_chunk = True
                    curr_content += delta

                    # édition périodique
                    if len(curr_content) >= 50:
                        text = (curr_content[-(max_message_length - len(STREAMING_INDICATOR)):] + STREAMING_INDICATOR)
                        await reply_helper_embed(text, final=False)

                if not got_any_chunk:
                    # aucun token reçu -> fallback message court
                    await reply_helper_embed("… (no content)", final=True)
                else:
                    await reply_helper_embed(curr_content, final=True)

                response_contents.append(curr_content or "")

            else:
                # Non-streaming (OpenRouter)
                resp = await openai_client.chat.completions.create(**openai_kwargs)
                content = (resp.choices[0].message.content or "").strip()
                if not content:
                    content = "*(No response content received from the model.)*"
                response_contents.append(content)
                await reply_helper_embed(content, final=True)

    except Exception:
        logging.exception("Error while generating response")

    # Enregistre le contenu généré dans le cache local
    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # GC du cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    token = resolve_bot_token(config)
    logging.info("Starting Discord bot with token from environment" if os.environ.get("DISCORD_BOT_TOKEN") else "Starting Discord bot with token from config.yaml")
    await discord_bot.start(token)


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
