from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup

player = playerid_lookup("skubal", "tarik")
player_id = player.iloc[0]["key_mlbam"]

data = statcast_pitcher("2025-04-01", "2025-04-15", player_id)

strikeouts = data[data["events"] == "strikeout"]

ks_by_game = strikeouts.groupby("game_date").size()

print("Strikeouts by game:")
print(ks_by_game)

print("Average Ks per game:", ks_by_game.mean())