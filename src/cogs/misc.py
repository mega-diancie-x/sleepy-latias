import time

import discord
from discord import app_commands
from discord.ext import commands

class MiscCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="ping", description="Ping the bot")
    async def ping(self, interaction: discord.Interaction):
        await interaction.response.send_message(f"Pong! Latency is {round(self.bot.latency * 1000)}ms")

    @app_commands.command(name="devecho", description="Send a message")
    async def devecho(self, interaction: discord.Interaction, message: str):
        if interaction.user.id != 977110493057658921:
            await interaction.response.send_message("You are not the developer of this bot.", ephemeral=True)
            return
        await interaction.response.send_message("Sent!", ephemeral=True)
        await interaction.channel.send(message)

    @app_commands.command(name="uptime", description="Get the bot's uptime")
    async def uptime(self, interaction: discord.Interaction):   
        uptime = time.time() - self.bot.start_time
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        await interaction.response.send_message(f"Uptime: {hours}h {minutes}m {seconds}s")

    @app_commands.command(name="help", description="Get a usage guide for the bot")
    async def help(self, interaction: discord.Interaction):
        await interaction.response.send_message("https://github.com/mega-diancie-x/sleepy-latias/blob/main/README.md")
                                                
    @app_commands.command(name="credits", description="Credits for the bot")
    async def credits(self, interaction: discord.Interaction):
        embed = discord.Embed(title="Credits",
              description="The people that made the bot possible",
              colour=0x00b0f4)
        embed.add_field(name="Developer",
        value="`@the_diancie`",
        inline=False)
        embed.add_field(name="Subskill Probability Distribution",
        value="[Raenonx](https://pks.raenonx.cc/en/docs/view/mechanics/subskill-probability)\n`@powderrrrrrrrrrrrrrrrrrrrrrrrrrr` and `@noko17` for data collection and parsing",
        inline=False)
        embed.add_field(name="Ingredient Combinations",
        value="[Raenonx](https://pks.raenonx.cc/en/docs/view/concept/ingredient-combo)\n`@calpisrtori` on Discord",
        inline=False)
        await interaction.response.send_message(embed=embed)

async def setup(bot: commands.Bot):
    await bot.add_cog(MiscCog(bot))