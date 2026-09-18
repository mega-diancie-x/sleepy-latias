# Sleepy Latias
A discord bot made to simulate Pokemon Sleep subskill probability.

## Usage Guide
The `/probability` command accepts **twelve** different parameters in total, with **three** being required.

Before proceeding, you should memorize the subskill and nature IDs using `/subskill_ids` and `/nature_ids`.  
To quickly remember Subskill IDs:  
BFS 0, HB 1,  
HSM 7, HSS 7 + 6  
IFM 8, IFS 8 + 6  
STM 9, STS 9 + 6  

### Basic Usage
The most basic setup for the command is providing only the three required parameters, with an example in the following:  
*Example 1*
```
/probability required_subskills: 0,1 species: Standard (5-7 pip) (10/40/100) allow_subseeds: true
```
The above command calculates the probability of getting the BFS+HB combo with subskills up to Lv.50 on a 5-7 pip Pokemon, with sub skill seeds being used when applicable.

#### Search Range
The `search_range` parameter specifies the Maximum Unlock Level of the subskills to be counted, from level 10 to 80.

#### Nature Specifications
The `nature_up` and `nature_down` parameters specify the nature of the Pokemon. The `nature_down_exclude` parameters specify which stat drops to avoid.  
*Example 2*
```
/probability required_subskills: 0,1 species: Standard (5-7 pip) (10/40/100) allow_subseeds: true nature_up: 0 nature_down: 3
```
This command calculates the probability with the same conditions as Example 1, except the Pokemon's nature must be Lonely.

#### Ingredient Combo
The ingredient combo of a pokemon can be specified with the `ingredients` parameter.

### Optional Subskills
You can specify optional subskills to be matched. Along with other conditions, a specified number or more of the optional subskills exists in the Search Range, that Pokemon satisfies the condition to be counted. Otherwise, the Pokemon counts as a fail.  
*Example 3*
```
/probability required_subskills: 7,9 species: 2 allow_subseeds: true optional_subskills: 13,15 optional_amount: 1 nature_up: 0,2 nature_down_exclude: 0,2 search_range: 3
```
This command calculates the probability of getting HSM+STM combo (subseeds allowed) AND **at least one of** HSS or STS at or before Level 50, with a SoH+ or MSC+ nature and neither stats being dropped by nature.

### Cumulative Probability
The bot allows the calculation of cumulative probability of a specified number of catches from a specified Friend Level (FL). The bot calciulates the probability of at least one Pokemon satisfying the conditions being in the captures.

#### Gold Cap
You can limit the maximum number of Guaranteed Golds being toggled on. This won't take effect before FL40.
