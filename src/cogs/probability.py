from random import random, randint, choice, choices
import numpy as np
from numba import njit, prange
from numba.cuda import atomic
import time
import asyncio

import discord
from discord import app_commands
from discord.app_commands import Choice
from discord.ext import commands

# Credits to Raenonx for data
GOLD_PROB = 0.14
BLUE_PROB = 0.33

# Define IDs and Names
ID_TO_SUBS = {
    0: "Berry Finding S", 1: "Helping Bonus", 2: "Dream Shard Bonus", 3: "Energy Recovery Bonus", 4: "Research EXP Bonus", 5: "Sleep EXP Bonus", 6: "Skill Level-Up M",
    7: "Helping Speed M", 8: "Ingredient Finder M", 9: "Skill Trigger M", 10: "Inventory Up M", 11: "Inventory Up L", 12: "Skill Level-Up S",
    13: "Helping Speed S", 14: "Ingredient Finder S", 15: "Skill Trigger S", 16: "Inventory Up S"
}

ID_TO_NATURES = {
    0: "Speed of Help", 1: "Ingredient Finding", 2: "Main Skill Chance", 3: "Energy Recovery", 4: "EXP Gains", 5: "None"
}

ID_TO_INGS = {
    111: "AAA", 112: "AAB", 113: "AAC", 121: "ABA", 122: "ABB", 123: "ABC"
}

TO_SUBSEED = {
    10: 11,
    12: 6,
    13: 7,
    14: 8,
    15: 9,
    16: 10
}

# Define NumPy Arrays for faster computation
to_subseed_keys = np.array(list(TO_SUBSEED.keys()))
to_subseed_values = np.array(list(TO_SUBSEED.values()))

GOLD_SUBS = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int8)
BLUE_SUBS = np.array([7, 8, 9, 10, 11, 12], dtype=np.int8)
WHITE_SUBS = np.array([13, 14, 15, 16], dtype=np.int8)
ALL_SUBS = np.concatenate((GOLD_SUBS, BLUE_SUBS, WHITE_SUBS))
NATURES = np.array([0, 1, 2, 3, 4], dtype=np.int8)

SPECIES_BADGE_LIST = {
    0: [10, 40, 100],
    1: [10, 30, 60],
    2: [10, 25, 50],
    3: [10, 20, 40]
}

masterDB = None
masterDB_task = None

# Change later if needed
SAMPLES_PER_BRACKET = 96_040_000
TOTAL_SAMPLES = SAMPLES_PER_BRACKET * 4

@njit(parallel=True)
def _generateMasterDB() -> np.ndarray:
    """
    Generates a master database of Pokemon Samples (including subskills, natures and ingredients)
    """
    db = np.empty((4, SAMPLES_PER_BRACKET, 8), dtype=np.int8)

    for golds in range(4):
        for i in prange(SAMPLES_PER_BRACKET):
            # Copy the arrays to avoid modifying the originals
            gold = np.empty(7, dtype=np.int8)
            blue = np.empty(6, dtype=np.int8)
            white = np.empty(4, dtype=np.int8)
            for _k in range(7): gold[_k] = GOLD_SUBS[_k]
            for _k in range(6): blue[_k] = BLUE_SUBS[_k]
            for _k in range(4): white[_k] = WHITE_SUBS[_k]
            # Subskill lengths
            gEnd = 7
            bEnd = 6
            wEnd = 4

            # Guaranteed Golds
            for _f in range(golds):
                idx = int(np.random.rand() * gEnd)
                db[golds, i, _f] = gold[idx]
                gold[idx] = gold[gEnd - 1]
                gEnd -= 1

            for j in range(golds, 5):
                # If all subskills are white, force a gold subskill
                if j == 4 and golds == 0:
                    allWhite = True
                    for k in range(4):
                        # White subskills are 13-16
                        if db[golds, i, k] < 13:
                            allWhite = False
                            break
                    if allWhite:
                        idx = int(np.random.rand() * gEnd)
                        db[golds, i, j] = gold[idx]
                        gold[idx] = gold[gEnd - 1]
                        gEnd -= 1
                        break

                # Choose a subskill based on probability
                r = np.random.rand()
                if r <= GOLD_PROB:
                    idx = int(np.random.rand() * gEnd)
                    db[golds, i, j] = gold[idx]
                    gold[idx] = gold[gEnd - 1]
                    gEnd -= 1
                elif r <= GOLD_PROB + BLUE_PROB:
                    idx = int(np.random.rand() * bEnd)
                    db[golds, i, j] = blue[idx]
                    blue[idx] = blue[bEnd - 1]
                    bEnd -= 1
                else:
                    idx = int(np.random.rand() * wEnd)
                    db[golds, i, j] = white[idx]
                    white[idx] = white[wEnd - 1]
                    wEnd -= 1

            # Natures
            n1 = int(np.random.rand() * 5)
            n2 = int(np.random.rand() * 5)
            if n1 == n2:
                db[golds, i, 5] = np.int8(5)
                db[golds, i, 6] = np.int8(5)
            else:
                db[golds, i, 5] = np.int8(n1)
                db[golds, i, 6] = np.int8(n2)

            # Ingredient speed
            ing = np.int8(10) if np.random.rand() < 1.0 / 3.0 else np.int8(20)
            db[golds, i, 7] = np.int8(100 + ing + 1 + int(np.random.rand() * 3))

    return db


def generateMasterDB() -> np.ndarray:
    # Helper function for time measurement
    start = time.time()
    db = _generateMasterDB()
    print(f"Master DB generated in {round(time.time() - start, 2)} seconds")
    return db

@njit(parallel=True)
def _queryDatabase(db, reqSubskills, optSubskills=None, optAmount=0, nature1=None, nature2=None, natureNot=None, ings=-1, searchRange: int = 3, allowSeeds=False):
    """
    Checks the database for matches to the given subskills and natures. Returns a list of hits for each bracket (4 total).
    """
    hits = np.zeros(4, dtype=np.int32)
    num_req = len(reqSubskills)
    max_idx = searchRange

    for bracket in range(4):
        for i in prange(SAMPLES_PER_BRACKET):
            if nature1 is not None:
                found = False
                for nn in range(len(nature1)):
                    if db[bracket, i, 5] == nature1[nn]:
                        found = True
                        break
                if not found:
                    continue
            if nature2 is not None:
                found = False
                for nn in range(len(nature2)):
                    if db[bracket, i, 6] == nature2[nn]:
                        found = True
                        break
                if not found:
                    continue
            if natureNot is not None:
                found = False
                for nn in range(len(natureNot)):
                    if db[bracket, i, 6] == natureNot[nn]:
                        found = True
                        break
                if found:
                    continue
            if ings != -1:
                if ings % 10 == 0:
                    if db[bracket, i, 7] // 10 != ings // 10:
                        continue
                else:
                    if db[bracket, i, 7] != ings:
                        continue

            req_found = 0
            used_slots = np.zeros(max_idx, dtype=np.bool_)

            for r_idx in range(num_req):
                req_target = reqSubskills[r_idx]
                matched = False

                for s_idx in range(max_idx):
                    if (not used_slots[s_idx]) and db[bracket, i, s_idx] == req_target:
                        used_slots[s_idx] = True
                        req_found += 1
                        matched = True
                        break
                if matched:
                    continue

                if allowSeeds:
                    k_idx = -1
                    for kk in range(len(to_subseed_values)):
                        if to_subseed_values[kk] == req_target:
                            k_idx = kk
                            break
                    if k_idx != -1:
                        base_skill = to_subseed_keys[k_idx]
                        present = False
                        for p in range(5):
                            if db[bracket, i, p] == req_target:
                                present = True
                                break
                        if not present:
                            for s_idx in range(max_idx):
                                if (not used_slots[s_idx]) and db[bracket, i, s_idx] == base_skill:
                                    used_slots[s_idx] = True
                                    req_found += 1
                                    matched = True
                                    break

                    if not matched:
                        present11 = False
                        for p in range(5):
                            if db[bracket, i, p] == 11:
                                present11 = True
                                break
                        if req_target == 11 and not present11:
                            for s_idx in range(max_idx):
                                if (not used_slots[s_idx]) and db[bracket, i, s_idx] == 16:
                                    used_slots[s_idx] = True
                                    req_found += 1
                                    matched = True
                                    break

            if req_found < num_req:
                continue

            if optSubskills is not None:
                opt_found = 0
                for o_idx in range(len(optSubskills)):
                    opt_target = optSubskills[o_idx]
                    matched = False

                    for s_idx in range(max_idx):
                        if (not used_slots[s_idx]) and db[bracket, i, s_idx] == opt_target:
                            used_slots[s_idx] = True
                            opt_found += 1
                            matched = True
                            break
                    if matched:
                        continue

                    if allowSeeds:
                        k_idx = -1
                        for kk in range(len(to_subseed_values)):
                            if to_subseed_values[kk] == opt_target:
                                k_idx = kk
                                break
                        if k_idx != -1:
                            base_skill = to_subseed_keys[k_idx]
                            present = False
                            for p in range(5):
                                if db[bracket, i, p] == opt_target:
                                    present = True
                                    break
                            if not present:
                                for s_idx in range(max_idx):
                                    if (not used_slots[s_idx]) and db[bracket, i, s_idx] == base_skill:
                                        used_slots[s_idx] = True
                                        opt_found += 1
                                        matched = True
                                        break

                        if not matched:
                            present11 = False
                            for p in range(5):
                                if db[bracket, i, p] == 11:
                                    present11 = True
                                    break
                            if opt_target == 11 and not present11:
                                for s_idx in range(max_idx):
                                    if (not used_slots[s_idx]) and db[bracket, i, s_idx] == 16:
                                        used_slots[s_idx] = True
                                        opt_found += 1
                                        matched = True
                                        break

                if opt_found < optAmount:
                    continue

            hits[bracket] += 1

    return hits


def queryDatabase(db, reqSubskills, optSubskills = None, optAmount = 0, nature1 = -1, nature2 = -1, natureNot = -1, ings = -1, searchRange: int = 3, allowSeeds = False):
    start = time.time()
    hits = _queryDatabase(db, reqSubskills, optSubskills, optAmount, nature1, nature2, natureNot, ings, searchRange, allowSeeds)
    print(f"Query took {round(time.time() - start, 2)} seconds")
    return hits

@njit(parallel=True)
def _score_query_db(db, subskill_scores, nature_up_scores, nature_down_scores, req_score, ings = -1, searchRange: int = 3, allowSeeds=False):
    scores = np.zeros(4, dtype=np.int64)
    hits = np.zeros(4, dtype=np.int32)
    max_idx = searchRange

    for bracket in range(4):
        for i in prange(SAMPLES_PER_BRACKET):
            if ings != -1:
                if ings % 10 == 0:
                    if db[bracket, i, 7] // 10 != ings // 10:
                        continue
                else:
                    if db[bracket, i, 7] != ings:
                        continue

            score = 0
            used_slots = np.zeros(max_idx, dtype=np.bool_)

            for s_idx in range(max_idx):
                val = db[bracket, i, s_idx]
                if allowSeeds:
                    present11 = False
                    for p in range(5):
                        if db[bracket, i, p] == 11:
                            present11 = True
                            break
                    if val == 16 and (not present11) and (not used_slots[s_idx]):
                        if subskill_scores[11] > subskill_scores[16]:
                            used_slots[s_idx] = True
                            score += subskill_scores[11]
                            continue

                    k_idx = -1
                    for kk in range(len(to_subseed_keys)):
                        if to_subseed_keys[kk] == val:
                            k_idx = kk
                            break
                    if k_idx != -1 and (not used_slots[s_idx]):
                        after_skill = to_subseed_values[k_idx]
                        present_after = False
                        for p in range(5):
                            if db[bracket, i, p] == after_skill:
                                present_after = True
                                break
                        if (not present_after) and (subskill_scores[after_skill] > subskill_scores[val]):
                            used_slots[s_idx] = True
                            score += subskill_scores[after_skill]
                            continue

                if not used_slots[s_idx]:
                    used_slots[s_idx] = True
                    score += subskill_scores[val]

            n5 = db[bracket, i, 5]
            n6 = db[bracket, i, 6]
            score += nature_up_scores[n5]
            score -= nature_down_scores[n6]

            scores[bracket] += score
            if score >= req_score:
                hits[bracket] += 1

    return [scores, hits]


def score_query_db(db, subskill_scores, nature_up_scores, nature_down_scores, req_score, ings = -1, searchRange: int = 3, allowSeeds=False):
    start = time.time()
    subskill_scores = np.array([subskill_scores.get(i, 0) for i in range(17)], dtype=np.int16)
    nature_up_scores = np.array([nature_up_scores.get(i, 0) for i in range(5)], dtype=np.int16)
    nature_down_scores = np.array([nature_down_scores.get(i, 0) for i in range(5)], dtype=np.int16)
    results = _score_query_db(db, subskill_scores, nature_up_scores, nature_down_scores, req_score, ings, searchRange, allowSeeds)
    print(f"Query took {round(time.time() - start, 2)} seconds")
    return results


def cumulative_probability(odds1: float, odds2: float, odds3: float, odds4: float, goldCap: int, species: int, trials: int, fl: int = 0) -> float:
    """
    Calculates the probability of getting AT LEAST ONE success across N trials, accounting for changing base odds when a new befriending medal is unlocked.
    """
    if trials <= 0:
        return 0.0

    bronzeBadge = SPECIES_BADGE_LIST[species][0]
    silverBadge = SPECIES_BADGE_LIST[species][1] - 1
    goldBadge = SPECIES_BADGE_LIST[species][2] - 1

    if goldCap == 1:
        silverBadge = 999999
        goldBadge = 999999
    elif goldCap == 2:
        goldBadge = 999999

    fail1 = 1.0 - odds1
    fail2 = 1.0 - odds2
    fail3 = 1.0 - odds3
    fail4 = 1.0 - odds4

    t1 = max(min(8, trials) - fl, 0)
    t2 = max(0, min(silverBadge - 9, trials - 9 - fl))
    t3 = max(0, min(goldBadge - silverBadge, trials - silverBadge + 1 - fl))
    if goldCap != 0:
        t4 = max(0, trials - goldBadge + 1 - fl)
    else:
        t3 = max(0, trials - silverBadge + 1 - fl)
        t4 = 0

    if goldCap != 0:
        total_fail_chance = (fail1 ** t1) * (fail2 ** t2) * (fail3 ** t3) * (fail4 ** t4)
    else:
        total_fail_chance = (fail1 ** t1) * (fail2 ** t2) * (fail1 ** t3)

    return 1.0 - total_fail_chance


def generateEmbed(hits, reqSubskills, optSubskills = None, optAmount = 0, nature1 = -1, nature2 = -1, natureNot = -1, ings = -1, cumulative = -1, goldCap = 3, species = 0, searchRange: int = 3, fl: int = 0):
    embed = discord.Embed(
        title="Probability Results",
        description="Notice: The bot is in development, and probability values may not be accurate.",
        color=discord.Color.random()
    )
    embed.add_field(name="Required Subskills", value="\n".join([ID_TO_SUBS[i] for i in reqSubskills]), inline=False)
    if optSubskills is not None:
        optStr = '\n'.join([ID_TO_SUBS[i] for i in optSubskills])
        embed.add_field(name="Optional Subskills", value=f"**{optAmount} of** \n{optStr}", inline=True)
    if nature1 is not None or nature2 is not None:
        natureStr = ""
        if nature1 is not None:
            natureStr += "<:nup:1522481221710647296> " + "/".join(ID_TO_NATURES[int(n)] for n in nature1)
        if nature2 is not None:
            natureStr += "\n<:ndown:1522481223815921734> " + "/".join(ID_TO_NATURES[int(n)] for n in nature2)
        embed.add_field(name="Required Nature", value=natureStr, inline=False)
    if natureNot is not None:
        natureNotStr = "<:ndown:1522481223815921734> " + "/".join(ID_TO_NATURES[int(n)] for n in natureNot) + " (Excluded)"
        embed.add_field(name="Excluded Nature", value=natureNotStr, inline=True)
    if ings != -1:
        embed.add_field(name="Required Ingredients", value=ID_TO_INGS.get(ings, str(ings)), inline=True)

    hitsStr = ""
    for i in range(4):
        hitsStr += f"{i} Guaranteed Golds"
        hitsStr += f"\n {hits[i]} Hits - {hits[i] / SAMPLES_PER_BRACKET * 100:.3f}% ± 0.01%\n"

    embed.add_field(name="Base Odds", value=hitsStr, inline=False)

    if cumulative != -1 and cumulative > 0:
        cumulativeStr = ""
        max_step = cumulative // 10
        for step in range(1, max_step + 1):
            catch = step * 10
            cumulativeStr += f"Odds at Catch #{catch + fl}: {cumulative_probability(hits[0] / SAMPLES_PER_BRACKET, hits[1] / SAMPLES_PER_BRACKET, hits[2] / SAMPLES_PER_BRACKET, hits[3] / SAMPLES_PER_BRACKET, goldCap, species, catch, fl) * 100 :.6f}%\n"
        if cumulative % 10 != 0:
            cumulativeStr += f"Odds at Catch #{cumulative + fl}: {cumulative_probability(hits[0] / SAMPLES_PER_BRACKET, hits[1] / SAMPLES_PER_BRACKET, hits[2] / SAMPLES_PER_BRACKET, hits[3] / SAMPLES_PER_BRACKET, goldCap, species, cumulative, fl) * 100 :.6f}%\n"
        embed.add_field(name="Cumulative Probability", value=cumulativeStr, inline=False)

    return embed


class ProbabilityCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="probability", description="Calculate the probability of a Pokemon having certain subskills")
    @app_commands.describe(
        required_subskills="Comma-separated subskill IDs that must all be present.",
        species="Species type to use for badge thresholds.",
        allow_subseeds="Whether subseed conversion is allowed.",
        optional_subskills="Optional subskill IDs that may also be present.",
        optional_amount="How many optional subskills are required.",
        nature_up="Nature IDs that must be boosted.",
        nature_down="Nature IDs that must be diminished.",
        nature_down_exclude="Nature IDs to exclude from being diminished.",
        ingredients="Ingredient combination requirement.",
        cumulative="Calculate cumulative odds up to this catch number.",
        gold_cap="Gold subskill max to use for cumulative calculation.",
        search_range="Number of unlocked subskill slots to search.",
        fl="Friendship level offset for cumulative calculation (Default: 0)"
    )
    @app_commands.choices(
        ingredients=[
            Choice(name="AAX (Lv60 any)", value=110),
            Choice(name="AAA (Mono)", value=111),
            Choice(name="AAB", value=112),
            Choice(name="AAC", value=113),
            Choice(name="ABA", value=121),
            Choice(name="ABB", value=122),
            Choice(name="ABC", value=123),
            Choice(name="None", value=-1)
        ],
        gold_cap=[
            Choice(name="0", value=0),
            Choice(name="1 (Bronze)", value=1),
            Choice(name="2 (Silver)", value=2),
            Choice(name="3 (Gold)", value=3)
        ],
        species=[
            Choice(name="Standard (5-7 pip) (10/40/100)", value=0),
            Choice(name="2nd Evos (10/30/60)", value=1),
            Choice(name="16 Pip (10/25/50)", value=2),
            Choice(name="20-30 Pip (10/20/40)", value=3)
        ],
        search_range=[
            Choice(name="Lv. 10", value=1),
            Choice(name="Lv. 25", value=2),
            Choice(name="Lv. 50", value=3),
            Choice(name="Lv. 70", value=4),
            Choice(name="Lv. 80", value=5)
        ]
    )
    async def probability(
        self,
        interaction: discord.Interaction,
        required_subskills: str,
        species: int,
        allow_subseeds: bool,
        optional_subskills: str = None,
        optional_amount: int = 1,
        nature_up: str = None,
        nature_down: str = None,
        nature_down_exclude: str = None,
        ingredients: int = -1,
        cumulative: int = -1,
        gold_cap: int = 3,
        search_range: int = 3,
        fl: int = 0
    ):
        if masterDB is None:
            await interaction.response.send_message("The probability database is still building. Please try again in a few minutes.", ephemeral=True)
            return

        if required_subskills == "":
            await interaction.response.send_message("You must specify at least one required subskill.", ephemeral=True)
            return
        if cumulative > 500000:
            await interaction.response.send_message("You can only calculate the cumulative probability for up to 500,000 catches.", ephemeral=True)
            return
        try:
            required_subskills = required_subskills.replace(" ", "")
            optional_subskills = optional_subskills.replace(" ", "") if optional_subskills is not None else None
            nature_up = nature_up.replace(" ", "") if nature_up is not None else None
            nature_down = nature_down.replace(" ", "") if nature_down is not None else None
            nature_down_exclude = nature_down_exclude.replace(" ", "") if nature_down_exclude is not None else None
        except AttributeError:
            pass

        for i in required_subskills.split(","):
            try:
                int(i)
            except ValueError:
                await interaction.response.send_message(f"Invalid subskill ID: {i}", ephemeral=True)
                return
            if int(i) not in ID_TO_SUBS:
                await interaction.response.send_message(f"Invalid subskill ID: {i}", ephemeral=True)
                return
        for i in optional_subskills.split(",") if optional_subskills is not None else []:
            try:
                int(i)
            except ValueError:
                await interaction.response.send_message(f"Invalid subskill ID: {i}", ephemeral=True)
                return
            if int(i) not in ID_TO_SUBS:
                await interaction.response.send_message(f"Invalid subskill ID: {i}", ephemeral=True)
                return
        for i in nature_up.split(",") if nature_up is not None else []:
            try:
                int(i)
            except ValueError:
                await interaction.response.send_message(f"Invalid nature ID: {i}", ephemeral=True)
                return
            if int(i) not in ID_TO_NATURES:
                await interaction.response.send_message(f"Invalid nature ID: {i}", ephemeral=True)
                return
        for i in nature_down.split(",") if nature_down is not None else []:
            try:
                int(i)
            except ValueError:
                await interaction.response.send_message(f"Invalid nature ID: {i}", ephemeral=True)
                return
            if int(i) not in ID_TO_NATURES:
                await interaction.response.send_message(f"Invalid nature ID: {i}", ephemeral=True)
                return
        for i in nature_down_exclude.split(",") if nature_down_exclude is not None else []:
            try:
                int(i)
            except ValueError:
                await interaction.response.send_message(f"Invalid nature ID: {i}", ephemeral=True)
                return
            if int(i) not in ID_TO_NATURES:
                await interaction.response.send_message(f"Invalid nature ID: {i}", ephemeral=True)
                return

        start = time.time()
        await interaction.response.defer()
        reqSubskills = np.array([int(i) for i in required_subskills.split(",")], dtype=np.int8)
        if optional_subskills is not None:
            optSubskills = np.array([int(i) for i in optional_subskills.split(",")], dtype=np.int8)
        else:
            optSubskills = None

        nature_up = np.array([int(i) for i in nature_up.split(",")], dtype=np.int8) if nature_up is not None else None
        nature_down = np.array([int(i) for i in nature_down.split(",")], dtype=np.int8) if nature_down is not None else None
        nature_down_exclude = np.array([int(i) for i in nature_down_exclude.split(",")], dtype=np.int8) if nature_down_exclude is not None else None

        results = queryDatabase(masterDB, reqSubskills, optSubskills, optional_amount, nature_up, nature_down, nature_down_exclude, ingredients, search_range, allow_subseeds)
        embed = generateEmbed(results, reqSubskills, optSubskills, optional_amount, nature_up, nature_down, nature_down_exclude, ingredients, cumulative, gold_cap, species, search_range, fl)
        embed.set_footer(text=f"Search Range: {search_range}, Total Samples: {TOTAL_SAMPLES}, Generated in {time.time() - start:.2f} seconds")
        await interaction.followup.send(embed=embed)
        print(f"Probability command executed in {time.time() - start:.2f} seconds")

    @app_commands.command(name="advanced_query", description="Query the database with custom scoring systems")
    @app_commands.describe(
        req_score="The minimum score needed for a match.",
        species="Species type to use for badge thresholds.",
        allow_subseeds="Whether subseed conversion is allowed.",
        subskill_scores="Comma-separated subskill score strings like 0010 for BFS with score 10.",
        nature_up_scores="Comma-separated nature-up score strings like 010 for Speed of Help +10.",
        nature_down_scores="Comma-separated nature-down score strings like 010 for a -10 penalty.",
        ings="Ingredient combination filter.",
        search_range="Number of unlocked subskill slots to search.",
        cumulative="Calculate cumulative odds up to this catch number.",
        gold_cap="Maximum number of guaranteed gold subskills to use for cumulative calculation.",
        fl="Friendship level offset for cumulative probability calculations."
    )
    @app_commands.choices(
        species=[
            Choice(name="Standard (5-7 pip) (10/40/100)", value=0),
            Choice(name="2nd Evos (10/30/60)", value=1),
            Choice(name="16 Pip (10/25/50)", value=2),
            Choice(name="20-30 Pip (10/20/40)", value=3)
        ],
        ings=[
            Choice(name="AAX (Lv60 any)", value=110),
            Choice(name="AAA (Mono)", value=111),
            Choice(name="AAB", value=112),
            Choice(name="AAC", value=113),
            Choice(name="ABA", value=121),
            Choice(name="ABB", value=122),
            Choice(name="ABC", value=123),
            Choice(name="None", value=-1)
        ],
        search_range=[
            Choice(name="Lv. 10", value=1),
            Choice(name="Lv. 25", value=2),
            Choice(name="Lv. 50", value=3),
            Choice(name="Lv. 70", value=4),
            Choice(name="Lv. 80", value=5)
        ]
    )
    async def advancedquery(
        self,
        interaction: discord.Interaction,
        req_score: int,
        species: int,
        allow_subseeds: bool,
        subskill_scores: str = "",
        nature_up_scores: str = "",
        nature_down_scores: str = "",
        ings: int = -1,
        search_range: int = 3,
        cumulative: int = -1,
        gold_cap: int = 3,
        fl: int = 0
    ):
        if masterDB is None:
            await interaction.response.send_message("The probability database is still building. Please try again in a few minutes.", ephemeral=True)
            return

        if subskill_scores == "" and nature_up_scores == "" and nature_down_scores == "":
            await interaction.response.send_message("You must specify at least one score.", ephemeral=True)
            return

        try:
            subskill_scores = subskill_scores.replace(" ", "")
            nature_up_scores = nature_up_scores.replace(" ", "")
            nature_down_scores = nature_down_scores.replace(" ", "")
        except AttributeError:
            pass

        try:
            subskill_scores = subskill_scores.split(",") if subskill_scores else [""]
            nature_up_scores = nature_up_scores.split(",") if nature_up_scores else [""]
            nature_down_scores = nature_down_scores.split(",") if nature_down_scores else [""]
            subScores = {}
            nupScores = {}
            ndownScores = {}

            for i in subskill_scores:
                if i == "":
                    continue
                if len(i) < 4 or len(i) > 5:
                    await interaction.response.send_message(f"Invalid subskill score formatting: {i}", ephemeral=True)
                    return
                id = int(i[:2])
                score = int(i[2:])
                if id not in ID_TO_SUBS:
                    await interaction.response.send_message(f"Invalid subskill ID: {i[:2]}", ephemeral=True)
                    return
                subScores[id] = score

            for i in nature_up_scores:
                if i == "":
                    continue
                if len(i) < 2 or len(i) > 3:
                    await interaction.response.send_message(f"Invalid nature up score formatting: {i}", ephemeral=True)
                    return
                id = int(i[0])
                score = int(i[1:])
                if id not in ID_TO_NATURES:
                    await interaction.response.send_message(f"Invalid nature ID: {i[0]}", ephemeral=True)
                    return
                nupScores[id] = score

            for i in nature_down_scores:
                if i == "":
                    continue
                if len(i) < 2 or len(i) > 3:
                    await interaction.response.send_message(f"Invalid nature down score formatting: {i}", ephemeral=True)
                    return
                id = int(i[0])
                score = int(i[1:])
                if id not in ID_TO_NATURES:
                    await interaction.response.send_message(f"Invalid nature ID: {i[0]}", ephemeral=True)
                    return
                ndownScores[id] = score
        except Exception:
            await interaction.response.send_message("Invalid score formatting.", ephemeral=True)
            return

        start = time.time()
        await interaction.response.defer()
        results = score_query_db(masterDB, subScores, nupScores, ndownScores, req_score, ings, search_range, allow_subseeds)
        scores = results[0]
        hits = results[1]
        embed = discord.Embed(
            title="Advanced Query Results",
            description="Notice: The bot is in development, and probability values may not be accurate.",
            color=discord.Color.random()
        )
        embed.add_field(name="Subskill Scores", value="\n".join([f"{ID_TO_SUBS[int(i)]}: {subScores[int(i)]}" for i in list(subScores.keys())]), inline=True)
        embed.add_field(name="Nature Up Scores", value="\n".join([f"<:nup:1522481221710647296> {ID_TO_NATURES[int(i)]}: {nupScores[int(i)]}" for i in list(nupScores.keys())]), inline=True)
        embed.add_field(name="Nature Down Scores", value="\n".join([f"<:ndown:1522481223815921734> {ID_TO_NATURES[int(i)]}: -{ndownScores[int(i)]}" for i in list(ndownScores.keys())]), inline=True)
        embed.add_field(name="Required Score", value=str(req_score), inline=True)
        scoresStr = ""
        for i in range(4):
            scoresStr += f"{i} Guaranteed Golds"
            scoresStr += f"\n {hits[i]} Hits - {hits[i] / SAMPLES_PER_BRACKET * 100:.3f}% ± 0.01%"
            scoresStr += f"\n Average Score: {scores[i] / SAMPLES_PER_BRACKET:.2f}\n"
        embed.add_field(name="Results", value=scoresStr, inline=False)
        if cumulative != -1 and cumulative > 0:
            cumulativeStr = ""
            max_step = cumulative // 10
            for step in range(1, max_step + 1):
                catch = step * 10
                cumulativeStr += f"Odds at Catch #{catch + fl}: {cumulative_probability(hits[0] / SAMPLES_PER_BRACKET, hits[1] / SAMPLES_PER_BRACKET, hits[2] / SAMPLES_PER_BRACKET, hits[3] / SAMPLES_PER_BRACKET, gold_cap, species, catch, fl) * 100 :.6f}%\n"
            if cumulative % 10 != 0:
                cumulativeStr += f"Odds at Catch #{cumulative + fl}: {cumulative_probability(hits[0] / SAMPLES_PER_BRACKET, hits[1] / SAMPLES_PER_BRACKET, hits[2] / SAMPLES_PER_BRACKET, hits[3] / SAMPLES_PER_BRACKET, gold_cap, species, cumulative, fl) * 100 :.6f}%\n"
            embed.add_field(name="Cumulative Probability", value=cumulativeStr, inline=False)
        embed.set_footer(text=f"Search Range: {search_range}, Total Samples: {TOTAL_SAMPLES}, Generated in {time.time() - start:.2f} seconds")
        await interaction.followup.send(embed=embed)
        print(f"Advanced Query command executed in {time.time() - start:.2f} seconds")

    @app_commands.command(name="query_format", description="Get the format for the advanced query command")
    async def queryformat(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="Query Format",
            description="The format for the advanced query command",
            color=discord.Color.random()
        )
        embed.add_field(name="Subskill Scores", value="`ID` (2 digits) + `Score` (Any digits) (e.g. `0010` for BFS with a score of 10)\nSeparate by comma (e.g. `0010,0110,0210`)", inline=False)
        embed.add_field(name="Nature Scores", value="`ID` (1 digit) + `Score` (Any digits) (e.g. `010` for SoH with a score of 10)\nSeparate by comma (e.g. `010,110,210`)", inline=False)
        embed.add_field(name="Nature Up/Down", value="Nature Up Scores are ADDED to the total score, while Nature Down Scores are SUBTRACTED from the total score.")
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="subskill_ids", description="Get the subskill IDs")
    async def subskillids(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="Subskill IDs",
            description="The IDs of the subskills",
            color=discord.Color.random()
        )
        for i in range(len(ID_TO_SUBS)):
            embed.add_field(name=ID_TO_SUBS[i], value="ID:" + str(i), inline=True)
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="nature_ids", description="Get the nature IDs")
    async def natureids(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="Nature IDs",
            description="The IDs of the natures",
            color=discord.Color.random()
        )
        for i in range(len(ID_TO_NATURES)):
            embed.add_field(name=ID_TO_NATURES[i], value="ID:" + str(i), inline=True)
        await interaction.response.send_message(embed=embed)

    """
    @app_commands.command(name="help_probability", description="Get help with the probability command")
    async def helpprobability(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="How to use `/probability`",
            description="`*` = Required Parameter\n\nTo quickly remember Subskill IDs:\nBFS 0, HB 1,\nHSM 7, HSS 7 + 6\nIFM 8, IFS 8 + 6\nSTM 9, STS 9 + 6\n\n__**Examples**__\n`/probability required_subskills:0,1 species:0 allow_subseeds:true`",
            color=0x00b0f4,
        )

        embed.add_field(name="*Required Subskills (`required_subskills`)", value="Enter comma-separated text of valid Subskill IDs (use `/subskill_ids` for IDs).", inline=False)
        embed.add_field(name="*Species Type (`species`)", value="The species type of the Pokemon based on its required friendship to catch. Should be selectable with values.", inline=False)
        embed.add_field(name="*Allow Subseeds (`allow_subseeds`)", value="Whether to allow the use of Sub Skill Seeds to reach the requirements or not.", inline=False)
        embed.add_field(name="Optional Subskills (`optional_subskills`)", value="Enter comma-separated text of valid Subskill IDs (use `/subskill_ids` for IDs). (Default: None)", inline=False)
        embed.add_field(name="Optional Subskill Count (`optional_amount`)", value="The number of optional subskills required. (Default: 1)", inline=False)
        embed.add_field(name="Nature Boost (`nature_up`)", value="The Nature ID(s) (comma-separated if multiple accepted) (use `/nature_ids` for IDs) required to be boosted by nature. (Default: Any)", inline=False)
        embed.add_field(name="Nature Diminish (`nature_down`)", value="The Nature ID(s) (comma-separated if multiple accepted) (use `/nature_ids` for IDs) required to be diminished by nature. (Default: Any)", inline=False)
        embed.add_field(name="Nature Diminish To Avoid (`nature_down_exclude`)", value="The Nature ID(s) (comma-separated if multiple needed) (use `/nature_ids` for IDs) to avoid being diminished by nature. (Default: All Accept)", inline=False)
        embed.add_field(name="Ingredient Combo (`ingredients`)", value="The Ingredient Combinations required. Should be selectable with values. (Default: None)", inline=False)
        embed.add_field(name="Cumulative Catches (`cumulative`)", value="The number of catches to calculate cumulative probability up to. Enter a number from 0 to 10000. (Default: None)", inline=False)
        embed.add_field(name="Gold Subskill Max (`gold_cap`)", value="The maximum number of guaranteed gold subskills to apply when calculating cumulative probability. Enter a number from 0 to 3. (Ignored before Silver Badge) (Default: 3)", inline=False)
        embed.add_field(name="Subskill Search Range (`search_range`)", value="The subskill slots to search up to. Should be selectable with values. (Default: Lv. 50)", inline=False)
        await interaction.response.send_message(embed=embed)
    """

    @app_commands.command(name="math_details", description="Get the math details of the probability calculation")
    async def mathdetails(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="Mathematical Details",
            description="The bot uses a Monte Carlo Simulation on the Pokemon Sleep Subskill selection process to acquire probability numbers.",
            color=discord.Color.random()
        )
        embed.add_field(name="Total Samples", value=str(TOTAL_SAMPLES), inline=False)
        embed.add_field(name="Samples per Bracket", value=str(SAMPLES_PER_BRACKET), inline=False)
        embed.add_field(name="Gold Probability", value=str(GOLD_PROB), inline=False)
        embed.add_field(name="Blue Probability", value=str(BLUE_PROB), inline=False)
        embed.add_field(name="White Probability", value=str(1 - GOLD_PROB - BLUE_PROB), inline=False)
        await interaction.response.send_message(embed=embed)


async def setup(bot: commands.Bot):
    await bot.add_cog(ProbabilityCog(bot))
    global masterDB_task

    async def _build():
        global masterDB
        masterDB = await asyncio.to_thread(generateMasterDB)
        print("Master DB build complete")

    masterDB_task = asyncio.create_task(_build())
