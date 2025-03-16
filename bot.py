# This example requires the 'message_content' intent.
import asyncio
import os
from asyncio import Lock, Queue

from discord.ext import commands, tasks
from dotenv import load_dotenv

from bot_funcs import *
from db import FeedbackDatabase

run_sentiment = get_sentiment_runner()
music_runner = get_music_runner()


def get_response(content: str, bot_mention: str) -> str:
    text = content.replace(bot_mention, '').lower().strip()
    if text == 'bass':
        return 'https://tenor.com/view/adamneelybass-adam-neely-adam-neely-bass-gif-18497855'
    sentiment: int = run_sentiment(text, bot_mention)
    if sentiment == 2:
        return 'Bass :)'
    elif sentiment == 1:
        return 'Bass'
    else:
        return 'BASS >:('


load_dotenv()

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.reactions = True

BOT_PREFIX: str = '!'
ONE_REACTION: str = '\u0031\ufe0f\u20E3'
TWO_REACTION: str = '\u0032\ufe0f\u20E3'

database: FeedbackDatabase = FeedbackDatabase()

bot: commands.Bot = commands.Bot(command_prefix=BOT_PREFIX, intents=intents)


@bot.command()
async def ping(ctx: commands.Context):
    await ctx.send('pong', reference=ctx.message)


@bot.command()
async def repeat(ctx: commands.Context, *, text: str):
    await ctx.send(text, reference=ctx.message)


music_gen_lock: Lock = Lock()


def _make_samples(prompt: str) -> tuple[tuple[Path, discord.File], tuple[Path, discord.File]]:
    one_path, one_file = music_runner(prompt, 'one')
    print('Done with one. Doing two now.')
    two_path, two_file = music_runner(prompt, 'two')
    return (one_path, one_file), (two_path, two_file)


queue_lock: Lock = Lock()
prompt_queue: Queue = Queue()


@bot.command('bass')
async def add_to_queue(ctx: commands.Context, *, prompt: str):
    await ctx.send('Adding you to queue...', reference=ctx.message)
    async with queue_lock:
        await prompt_queue.put((ctx, prompt))
    await ctx.send(f'Queue length is {prompt_queue.qsize()}.')


async def make_bass(ctx: commands.Context, *, prompt: str):
    try:
        async with music_gen_lock:
            await ctx.send('Running your prompt now chief!', reference=ctx.message)
            (one_path, one_file), (two_path, two_file) = await asyncio.to_thread(lambda: _make_samples(prompt))
        msg = await ctx.send('Here you go boss!\nWhich did you like more? 1 or 2?', files=[one_file, two_file],
                             reference=ctx.message)
        await msg.add_reaction(ONE_REACTION)
        await msg.add_reaction(TWO_REACTION)
        database.create_new_feedback(msg.id, ctx.author.id)
        database.create_prompt(msg.id, prompt, one_path, two_path)
    except ValueError as e:
        await ctx.send(f'Sorry king the mixtape was too fire. Error: {e}', reference=ctx.message)
    except FileNotFoundError as e:
        await ctx.send(f'Could not save music sample to database. Yell at Trystan. Error: {e}', reference=ctx.message)
    except Exception as e:
        await ctx.send(f'Sorry boss but something went wrong. Error: {e}', reference=ctx.message)


@bot.command('queue')
async def queue_len(ctx: commands.Context):
    # Gets the current queue length
    is_running: bool = music_gen_lock.locked()
    num_in_line: int = prompt_queue.qsize()
    running_mod: str = 'not ' if is_running else ''
    await ctx.send(f'Queue length is {num_in_line} prompts and we are {running_mod}currently running something.',
                   reference=ctx.message)


@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    message_id: int = payload.message_id
    if payload.user_id == bot.user.id:
        return
    if not database.contains_message(message_id):
        return
    if payload.emoji.name not in {ONE_REACTION, TWO_REACTION}:
        return
    is_one: bool = payload.emoji.name == ONE_REACTION
    database.update_feedback(message_id, payload.user_id, is_one)
    print(f'Got feedback for Message #{message_id}. One was preferred? {is_one}')


# @bot.command('single')
# async def make_single_bass(ctx: commands.Context, *, prompt: str):
#     await ctx.send('Please give me a bit...', reference=ctx.message)
#     try:
#         if music_gen_lock.locked():
#             await add_to_queue(ctx, prompt)
#             await ctx.send(
#                 f'[BUSY PROCESSING OTHER REQUESTS] You\'ll need to wait your turn for a bit... Queue length is {prompt_queue.qsize()}',
#                 reference=ctx.message)
#             return
#         async with music_gen_lock:
#             _, wav_file = await asyncio.to_thread(lambda: music_runner(prompt, None))
#         await ctx.send('Here you go boss!', file=wav_file, reference=ctx.message)
#     except ValueError as e:
#         await ctx.send(f'Sorry king the mixtape was too fire. Error: {e}', reference=ctx.message)
#     except Exception as e:
#         await ctx.send(f'Sorry boss but something went wrong. Error: {e}', reference=ctx.message)


@bot.event
async def on_message(message: discord.Message):
    if message.content.startswith(BOT_PREFIX):
        return await bot.process_commands(message)
    if message.author != bot.user and bot.user in message.mentions:
        response: str = get_response(message.content, bot.user.mention)
        await message.channel.send(response, reference=message)


@tasks.loop(seconds=0.5, name='queue_processor')
async def process_queue():
    if not prompt_queue.empty():
        print(f'Processing queue: {prompt_queue.qsize()}')
        async with queue_lock:
            print('Getting from queue')
            context, prompt = await prompt_queue.get()
        await make_bass(context, prompt=prompt)


@bot.event
async def on_ready():
    process_queue.start()


bot.run(os.environ['DISCORD_TOKEN'])
# @bot.command(name='help')
# async def _help(ctx: commands.Context):
#     pass

# client = MyClient(intents=intents)
# client.run(os.environ['DISCORD_TOKEN'])
