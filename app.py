import streamlit as st
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup

st.title("MLB Strikeout Prop Predictor")

last_name = st.text_input("Pitcher Last Name")
first_name = st.text_input("Pitcher First Name")

if st.button("Predict"):

    player = playerid_lookup(last_name, first_name)

    if player.empty:
        st.error("Player not found")

    else:
        player_id = player.iloc[0]["key_mlbam"]

        data = statcast_pitcher(
            "2025-04-01",
            "2025-04-15",
            player_id
        )

        strikeouts = data[data["events"] == "strikeout"]

        ks_by_game = strikeouts.groupby("game_date").size()

        avg_ks = ks_by_game.mean()

        st.subheader("Prediction")

        st.write(f"Average Ks Per Game: {avg_ks:.2f}")

        if avg_ks >= 7:
            st.success("Strong Strikeout Pitcher")
        else:
            st.warning("Lower Strikeout Projection")