# This example requires the 'message_content' intent.

from datetime import datetime
import os
from pathlib import Path
import typing as tp

from asyncio import Lock
import discord
from discord.ext import commands
import numpy as np
import torch.cuda
from dotenv import load_dotenv
from scipy.special import softmax
import scipy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_sentiment_runner() -> tp.Callable[[str, str], int]:
    task = 'sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL, device_map=get_device())
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, device_map=get_device())

    def inner(text: str, bot_mention: str) -> int:
        text = text.replace(bot_mention, '').lower().strip()
        print(f'Getting sentiment for "{text}"')
        encoded_input = tokenizer(text, return_tensors='pt').to(get_device())
        output = model(**encoded_input)
        scores: np.ndarray = output[0][0].detach().cpu().numpy()
        scores = softmax(scores)
        rankings: np.ndarray = np.argsort(scores)
        return int(rankings[-1])
    print('Set up sentiment runner')
    return inner

run_sentiment = get_sentiment_runner()


def get_music_runner() -> tp.Callable[[str], discord.File]:
    model: str = 'facebook/musicgen-large'
    synthesiser = pipeline("text-to-audio", model=model, device=get_device())

    def prompt_to_filename(prompt: str) -> str:
        curr_time: str = datetime.now().strftime("%Y%m%d-%H%M%S")
        return curr_time + '_'.join(prompt.lower().split(' ')[:5]) + '.wav'

    def inner(prompt: str) -> discord.File:
        filename: str = prompt_to_filename(prompt)
        sample = synthesiser(prompt, forward_params={'do_sample': True})
        actual_audio: np.ndarray = sample['audio'][0, 0]
        sampling_rate = sample['sampling_rate']
        scipy.io.wavfile.write(filename, rate=sampling_rate, data=actual_audio)
        return discord.File(filename)
    print('Set up music runner')
    return inner


music_runner = get_music_runner()


def get_response(content: str, bot_mention: str) -> str:
    sentiment: int = run_sentiment(content, bot_mention)
    if sentiment == 2:
        return 'Bass :)'
    elif sentiment == 1:
        return 'Bass'
    else:
        return 'BASS >:('


class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        print(f'Message from {message.author}: {message.content}')
        if message.author != self.user and self.user in message.mentions:
            response: str = get_response(message.content, self.user.mention)
            await message.channel.send(response)


load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

BOT_PREFIX: str = '!'

bot: commands.Bot = commands.Bot(command_prefix=BOT_PREFIX, intents=intents)

@bot.command()
async def ping(ctx: commands.Context):
    await ctx.send('pong', reference=ctx.message)


@bot.command()
async def repeat(ctx: commands.Context, *, text: str):
    await ctx.send(text, reference=ctx.message)

music_gen_lock: Lock = Lock()

@bot.command('bass')
async def make_bass(ctx: commands.Context, *, prompt: str):
    await ctx.send('Please give me a bit...', reference=ctx.message)
    try:
        if music_gen_lock.locked():
            await ctx.send('[BUSY PROCESSING OTHER REQUESTS] You\'ll need to wait your turn for a bit...', reference=ctx.message)
        async with music_gen_lock:
            wav_file: discord.File = music_runner(prompt)
        await ctx.send('Here you go boss!', file=wav_file, reference=ctx.message)
    except ValueError as e:
        await ctx.send(f'Sorry king the mixtape was too fire. Error: {e}', reference=ctx.message)
    except Exception as e:
        await ctx.send(f'Sorry boss but something went wrong. Error: {e}', reference=ctx.message)


@bot.event
async def on_message(message: discord.Message):
    if message.content.startswith(BOT_PREFIX):
        return await bot.process_commands(message)
    if message.author != bot.user and bot.user in message.mentions:
        response: str = get_response(message.content, bot.user.mention)
        await message.channel.send(response, reference=message)

bot.run(os.environ['DISCORD_TOKEN'])
# @bot.command(name='help')
# async def _help(ctx: commands.Context):
#     pass

# client = MyClient(intents=intents)
# client.run(os.environ['DISCORD_TOKEN'])