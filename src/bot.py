import os
import json
import traceback
import asyncio

import discord
from discord.ext import tasks, commands
from discord import app_commands

from dotenv import load_dotenv

COGS = [
    "cogs.simulation",
    "cogs.probability",
    "cogs.misc"
]

load_dotenv()

intents = discord.Intents.default()

class SleepyLatias(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="s!", intents=intents, description="Pokemon Sleep Subskill and Research Simulation")
        # self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        for cog in COGS:
            await self.load_extension(cog)
            print(f"Loaded {cog}")
        synced = await self.tree.sync()
        print("Synced", len(synced), "commands.")

    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")

    async def close(self):
        await super().close()

bot = SleepyLatias()

@bot.command()
async def sync(ctx: commands.Context):
    if ctx.author.id != 977110493057658921:
        await ctx.send("You are not the developer of this bot.", ephemeral=True)
        return
    synced = await bot.tree.sync()
    print(f"Synced {len(synced)} commands.")
    await ctx.send(f"Synced {len(synced)} commands.")

@bot.tree.error
async def on_tree_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    tb_str = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
    prefix = f"Sorry! An error occurred during {interaction.command.name}: \n```"
    suffix = "```"
    max_tb = 2000 - len(prefix) - len(suffix)
    if len(tb_str) > max_tb:
        tb_str = tb_str[:max_tb]
    msg = f"{prefix}{tb_str}{suffix}"
    if interaction.response.is_done():
        await interaction.followup.send(msg)
    else:
        await interaction.response.send_message(msg)
    print(tb_str)

async def main():
    token = os.getenv("TOKEN")
    try:
        await bot.start(token)
    except Exception as error:
        print(error)
        await bot.close()
        raise error

if __name__ == "__main__":
    asyncio.run(main())