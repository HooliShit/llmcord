import asyncio
from datetime import datetime as dt
import logging
from os import environ as env

import discord
from dotenv import load_dotenv
from litellm import acompletion

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

LOCAL_LLM: bool = env["LLM"].startswith("local/")
VISION_LLM: bool = "gpt-4-vision-preview" in env["LLM"]
MAX_COMPLETION_TOKENS = int(env.get('MAX_COMPLETION_TOKENS', 1024))
TEMPERATURE = float(env.get('TEMPERATURE', 1.0))
TOP_P = float(env.get('TOP_P', 1.0))

ALLOWED_CHANNEL_TYPES = (discord.ChannelType.text, discord.ChannelType.public_thread, discord.ChannelType.private_thread, discord.ChannelType.private)
ALLOWED_CHANNEL_IDS = tuple(int(id) for id in env["ALLOWED_CHANNEL_IDS"].split(",") if id)
ALLOWED_ROLE_IDS = tuple(int(id) for id in env["ALLOWED_ROLE_IDS"].split(",") if id)
MAX_IMAGES = int(env["MAX_IMAGES"]) if VISION_LLM else 0
MAX_MESSAGES = int(env["MAX_MESSAGES"])

EMBED_COLOR = {"incomplete": discord.Color.orange(), "complete": discord.Color.green()}
EMBED_MAX_LENGTH = 4096
EDITS_PER_SECOND = 1.3
MAX_MESSAGE_NODES = 100

if env["DISCORD_CLIENT_ID"]:
    print(f"\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={env['DISCORD_CLIENT_ID']}&permissions=412317273088&scope=bot\n")

system_prompt_extras = []
if any(env["LLM"].startswith(x) for x in ("gpt", "openai/gpt")) and "gpt-4-vision-preview" not in env["LLM"]:
    system_prompt_extras.append("User's names are their Discord IDs and should be typed as '<@ID>'.")

extra_kwargs = {}
if LOCAL_LLM:
    env["LLM"] = env["LLM"].replace("local/", "", 1)
    extra_kwargs["base_url"] = env["LOCAL_SERVER_URL"]
    extra_kwargs["api_key"] = env["LOCAL_API_KEY"] or "Not used"
    if env["OOBABOOGA_CHARACTER"]:
        extra_kwargs["mode"] = "chat"
        extra_kwargs["character"] = env["OOBABOOGA_CHARACTER"]

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=env["CUSTOM_DISCORD_STATUS"][:128] or "github.com/jakobdylanc/discord-llm-chatbot")
discord_client = discord.Client(intents=intents, activity=activity)

msg_nodes = {}
msg_locks = {}
last_task_time = None


class MsgNode:
    def __init__(self, data, too_many_images=False, fetch_next_failed=False, replied_to_msg=None):
        self.data = data
        self.too_many_images: bool = too_many_images
        self.fetch_next_failed: bool = fetch_next_failed
        self.replied_to_msg = replied_to_msg


def get_system_prompt():
    return [
        {
            "role": "system",
            "content": "\n".join([env["CUSTOM_SYSTEM_PROMPT"]] + system_prompt_extras + [f"Today's date: {dt.now().strftime('%B %d %Y')}"]),
        }
    ]


@discord_client.event
async def on_message(msg):
    global msg_nodes, msg_locks, last_task_time

    # Filter out unwanted messages
    if (
        msg.channel.type not in ALLOWED_CHANNEL_TYPES
        or (msg.channel.type != discord.ChannelType.private and discord_client.user not in msg.mentions)
        or (ALLOWED_CHANNEL_IDS and not any(id in ALLOWED_CHANNEL_IDS for id in (msg.channel.id, getattr(msg.channel, "parent_id", None))))
        or (ALLOWED_ROLE_IDS and (msg.channel.type == discord.ChannelType.private or not any(role.id in ALLOWED_ROLE_IDS for role in msg.author.roles)))
        or msg.author.bot
    ):
        return

    # Build message reply chain and set user warnings
    reply_chain = []
    user_warnings = set()
    curr_msg = msg
    while curr_msg and len(reply_chain) < MAX_MESSAGES:
        async with msg_locks.setdefault(curr_msg.id, asyncio.Lock()):
            if curr_msg.id not in msg_nodes:
                curr_msg_role = "assistant" if curr_msg.author == discord_client.user else "user"
                curr_msg_content = (curr_msg.embeds[0].description if curr_msg.embeds and curr_msg.author.bot else curr_msg.content) or " "
                if curr_msg_content.startswith(discord_client.user.mention):
                    curr_msg_content = curr_msg_content.replace(discord_client.user.mention, "", 1).lstrip() or " "
                curr_msg_images = [att for att in curr_msg.attachments if "image" in att.content_type]
                if VISION_LLM:
                    curr_msg_content = [{"type": "text", "text": curr_msg_content}]
                    curr_msg_content += [{"type": "image_url", "image_url": {"url": img.url, "detail": "low"}} for img in curr_msg_images[:MAX_IMAGES]]
                msg_nodes[curr_msg.id] = MsgNode(
                    {
                        "role": curr_msg_role,
                        "content": curr_msg_content,
                        "name": str(curr_msg.author.id),
                    },
                    too_many_images=len(curr_msg_images) > MAX_IMAGES,
                )

                try:
                    if (
                        not curr_msg.reference
                        and discord_client.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and any(prev_msg_in_channel.type == type for type in (discord.MessageType.default, discord.MessageType.reply))
                        and prev_msg_in_channel.author == curr_msg.author
                    ):
                        msg_nodes[curr_msg.id].replied_to_msg = prev_msg_in_channel
                    else:
                        next_is_thread_parent: bool = not curr_msg.reference and curr_msg.channel.type == discord.ChannelType.public_thread
                        if next_msg_id := curr_msg.channel.id if next_is_thread_parent else getattr(curr_msg.reference, "message_id", None):
                            while msg_locks.setdefault(next_msg_id, asyncio.Lock()).locked():
                                await asyncio.sleep(0)
                            msg_nodes[curr_msg.id].replied_to_msg = (
                                (curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(next_msg_id))
                                if next_is_thread_parent
                                else (r if isinstance(r := curr_msg.reference.resolved, discord.Message) else await curr_msg.channel.fetch_message(next_msg_id))
                            )
                except (discord.NotFound, discord.HTTPException, AttributeError):
                    logging.exception("Error fetching next message in the chain")
                    msg_nodes[curr_msg.id].fetch_next_failed = True

            curr_node = msg_nodes[curr_msg.id]
            reply_chain += [curr_node.data]
            if curr_node.too_many_images:
                user_warnings.add(f"⚠️ Max {MAX_IMAGES} image{'' if MAX_IMAGES == 1 else 's'} per message" if MAX_IMAGES > 0 else "⚠️ Can't see images")
            if curr_node.fetch_next_failed or (curr_node.replied_to_msg and len(reply_chain) == MAX_MESSAGES):
                user_warnings.add(f"⚠️ Only using last{'' if (count := len(reply_chain)) == 1 else f' {count}'} message{'' if count == 1 else 's'}")
            curr_msg = curr_node.replied_to_msg

    logging.info(f"Message received: {reply_chain[0]}, reply chain length: {len(reply_chain)}")

    # Generate and send response message(s) (can be multiple if response is long)
    response_msgs = []
    response_contents = []
    prev_content = None
    edit_task = None
    kwargs = dict(model=env["LLM"], messages=(get_system_prompt() + reply_chain[::-1]), temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_COMPLETION_TOKENS, stream=True) | extra_kwargs
    try:
        async with msg.channel.typing():
            async for chunk in await acompletion(**kwargs):
                curr_content = chunk.choices[0].delta.content or ""
                if prev_content:
                    if not response_msgs or len(response_contents[-1] + prev_content) > EMBED_MAX_LENGTH:
                        reply_to_msg = msg if not response_msgs else response_msgs[-1]
                        embed = discord.Embed(description="⏳", color=EMBED_COLOR["incomplete"])
                        for warning in sorted(user_warnings):
                            embed.add_field(name=warning, value="", inline=False)
                        response_msgs += [
                            await reply_to_msg.reply(
                                embed=embed,
                                silent=True,
                            )
                        ]
                        await msg_locks.setdefault(response_msgs[-1].id, asyncio.Lock()).acquire()
                        last_task_time = dt.now().timestamp()
                        response_contents += [""]

                    response_contents[-1] += prev_content
                    is_final_edit: bool = chunk.choices[0].finish_reason or len(response_contents[-1] + curr_content) > EMBED_MAX_LENGTH
                    if is_final_edit or (not edit_task or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDITS_PER_SECOND:
                        while edit_task and not edit_task.done():
                            await asyncio.sleep(0)
                        if response_contents[-1].strip():
                            embed.description = response_contents[-1]
                        embed.color = EMBED_COLOR["complete"] if is_final_edit else EMBED_COLOR["incomplete"]
                        edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                        last_task_time = dt.now().timestamp()

                prev_content = curr_content
    except:
        logging.exception("Error while streaming response")

    # Create MsgNodes for response messages
    for response_msg in response_msgs:
        msg_nodes[response_msg.id] = MsgNode(
            {
                "role": "assistant",
                "content": "".join(response_contents) or " ",
                "name": str(discord_client.user.id),
            },
            replied_to_msg=msg,
        )
        msg_locks[response_msg.id].release()

    # Delete MsgNodes for oldest messages (lowest IDs)
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_locks.setdefault(msg_id, asyncio.Lock()):
                msg_nodes.pop(msg_id, None)
                msg_locks.pop(msg_id, None)


async def main():
    await discord_client.start(env["DISCORD_BOT_TOKEN"])


asyncio.run(main())
