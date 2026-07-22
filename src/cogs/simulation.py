import os
import copy
from random import random, randint, choice, choices
from typing import Optional, Literal
import traceback

import discord
from discord import app_commands
from discord.ext import commands

GOLD_SUBSKILLS = [
    "Berry Finding S",
    "Helping Bonus",
    "Dream Shard Bonus",
    "Energy Recovery Bonus",
    "Research EXP Bonus",
    "Sleep EXP Bonus",
    "Skill Level-Up M"
]

BLUE_SUBSKILLS = [
    "Helping Speed M",
    "Ingredient Finder M",
    "Skill Trigger M",
    "Inventory Up M",
    "Inventory Up L",
    "Skill Level-Up S"
]

WHITE_SUBSKILLS = [
    "Helping Speed S",
    "Ingredient Finder S",
    "Skill Trigger S",
    "Inventory Up S"
]

NATURES = [
    "Speed of Help",
    "Ingredient Finding",
    "Main Skill Chance",
    "Energy Recovery",
    "EXP Gains"
]

SEEDABLE_SUBSKILLS = {
    "Helping Speed M": "Helping Speed S",
    "Ingredient Finder M": "Ingredient Finder S",
    "Skill Trigger M": "Skill Trigger S",
    "Inventory Up L": "Inventory Up M",
    "Inventory Up M": "Inventory Up S",
    "Skill Level-Up M": "Skill Level-Up S"
}

class Pokemon:
    def __init__(self):
        self.ingredients = ""
        self.subskills = []
        self.nature1 = ""
        self.nature2 = ""
        self.shiny = (randint(1, 450) == 450)

    def sim(self, badge: str = "N/A"):
        goldSubs = copy.deepcopy(GOLD_SUBSKILLS)
        blueSubs = copy.deepcopy(BLUE_SUBSKILLS)
        whiteSubs = copy.deepcopy(WHITE_SUBSKILLS)
        self.ingredients = "A" + choice(["A", "B", "B"]) + choice(["A", "B", "C"])
        if badge == "Bronze":
            self.subskills.append(choice(goldSubs))
            goldSubs.remove(self.subskills[0])
        elif badge == "Silver":
            for i in range(2):
                self.subskills.append(choice(goldSubs))
                goldSubs.remove(self.subskills[i])
        elif badge == "Gold":
            for i in range(3):
                self.subskills.append(choice(goldSubs))
                goldSubs.remove(self.subskills[i])
        while len(self.subskills) < 5:
            if len(self.subskills) == 4:
                allWhite = True
                for subskill in self.subskills:
                    if subskill not in WHITE_SUBSKILLS:
                        allWhite = False
                if allWhite:
                    self.subskills.append(choice(goldSubs))
                    goldSubs.remove(self.subskills[-1])
                    break
            subskillChoice = random()
            if subskillChoice <= 0.14:
                self.subskills.append(choice(goldSubs))
                goldSubs.remove(self.subskills[-1])
            elif subskillChoice <= 0.47:
                self.subskills.append(choice(blueSubs))
                blueSubs.remove(self.subskills[-1])
            else:
                self.subskills.append(choice(whiteSubs))
                whiteSubs.remove(self.subskills[-1])
        self.nature1 = choice(NATURES)
        self.nature2 = choice(NATURES)
        if self.nature1 == self.nature2:
            self.nature1 = "None"
            self.nature2 = "None"
        return [self.ingredients, self.subskills, self.nature1, self.nature2]

    def toStr(self, ingredients, subskills, nature1, nature2):
        if nature1 == "None" and nature2 == "None":
            nature = "This nature has no effect"
        else:
            nature =  "<:nup:1522481221710647296> " + nature1 + "\n<:ndown:1522481223815921734> " + nature2
        return [ingredients, "\n".join(subskills), nature]

    def toShortStr(self, ingredients, subskills, nature1, nature2):
        for i in range(len(subskills)):
            if subskills[i] == "Berry Finding S": subskills[i] = "BFS"
            if subskills[i] == "Helping Bonus": subskills[i] = "HB"
            if subskills[i] == "Dream Shard Bonus": subskills[i] = "DSB"
            if subskills[i] == "Energy Recovery Bonus": subskills[i] = "ERB"
            if subskills[i] == "Research EXP Bonus": subskills[i] = "RExp"
            if subskills[i] == "Sleep EXP Bonus": subskills[i] = "SExp"
            if subskills[i] == "Skill Level-Up M": subskills[i] = "SLUM"
            if subskills[i] == "Helping Speed M": subskills[i] = "HSM"
            if subskills[i] == "Ingredient Finder M": subskills[i] = "IFM"
            if subskills[i] == "Skill Trigger M": subskills[i] = "STM"
            if subskills[i] == "Inventory Up M": subskills[i] = "IUM"
            if subskills[i] == "Inventory Up L": subskills[i] = "IUL"
            if subskills[i] == "Skill Level-Up S": subskills[i] = "SLUS"
            if subskills[i] == "Helping Speed S": subskills[i] = "HSS"
            if subskills[i] == "Ingredient Finder S": subskills[i] = "IFS"
            if subskills[i] == "Skill Trigger S": subskills[i] = "STS"
            if subskills[i] == "Inventory Up S": subskills[i] = "IUS"
        
        if nature1 == "None" and nature2 == "None":
            nature = "This nature has no effect"
        else:
            nature =  "<:nup:1522481221710647296> " + nature1 + " <:ndown:1522481223815921734> " + nature2
            nature = nature.replace("Speed of Help", "SoH")
            nature = nature.replace("Ingredient Finding", "Ing")
            nature = nature.replace("Main Skill Chance", "MSC")
            nature = nature.replace("Energy Recovery", "Energy")
            nature = nature.replace("EXP Gains", "EXP")
        return [ingredients, ",".join(subskills), nature]

    def searchForSub(self, subskill: str, range: int = 3):
        return subskill in self.subskills[:range]

    def searchForSeedable(self, subskill: str, range: int = 3):
        if subskill not in list(SEEDABLE_SUBSKILLS.keys()):
            return subskill in self.subskills[:range]
        if subskill in self.subskills[:range]:
            return True
        if SEEDABLE_SUBSKILLS[subskill] in self.subskills[:range]:
            return subskill not in self.subskills
    
class SimulationCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="simulate", description="Simulate a Pokemon's rolls")
    @app_commands.describe(badge="The Befriending Badge you have", count="The number of Pokemon to simulate (Default: 1)", use_short_form="Use a shorter form of the output (Forced with more than 10 Pokemon) (Default: False)", name="The name of the Pokemon (Default: Pokemon)")
    async def simulate(self, interaction: discord.Interaction, badge: Literal["N/A", "Bronze", "Silver", "Gold"] = "N/A", count: int = 1, use_short_form: bool = False, name: str = "Pokemon"):
        if count > 50:
            await interaction.response.send_message("You can only simulate up to 50 Pokemon at a time.", ephemeral=True)
            return
        elif count > 10:
            use_short_form = True
            
        await interaction.response.defer()
        for i in range(count):
            pokemon = Pokemon()
            if use_short_form:
                results = pokemon.toShortStr(*pokemon.sim(badge))
            else:
                results = pokemon.toStr(*pokemon.sim(badge))
            embed = discord.Embed(
                title=f"{'✨ Shiif pokemon.shiny else ''}{name} #{i+1}",
                description="The results of the simulation",
                color=discord.Color.random()
            )
            embed.add_field(name="Ingredients", value=results[0], inline=False)
            embed.add_field(name="Subskills", value=results[1], inline=False)
            embed.add_field(name="Nature", value=results[2], inline=False)
            await interaction.followup.send(embed=embed)

async def setup(bot: commands.Bot):
    await bot.add_cog(SimulationCog(bot))