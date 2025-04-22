import streamlit as st
import pandas as pd
import gdown
import os
from recommender import create_song_features, recommend_songs
from user_recommender import get_user_top_songs, recommend_songs_for_user

# Load data
music_df = pd.read_csv("music_df_cleaned.csv")

GOOGLE_DRIVE_FILE_ID = "147yuIpQ74JCCkErdq4KZncwUeBvmo5rt"
user_history_file = "User Listening History.csv"

if not os.path.exists(user_history_file):
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, user_history_file, quiet=False)

user_history_df = pd.read_csv(user_history_file)


# Feature generation
music_df, numeric_features, text_features, tfidf = create_song_features(music_df)

# ------------------ UI ------------------

st.set_page_config(page_title="Smart Music Recommender", layout="wide")

# ----- Sidebar -----
with st.sidebar:
    st.title("Settings")

    rec_type = st.radio("Recommendation Type", ["Content-based", "User-based"])

    # Define sim_weight globally for both modes
    sim_weight = st.slider(
        "Similarity Preference: Audio vs. Genre/Tags",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    if rec_type == "Content-based":
        artist_list = sorted(music_df['artist'].dropna().unique())
        selected_artist = st.selectbox("Filter by Artist", ["All"] + artist_list)

    else:  # User-based
        user_history_df = user_history_df.dropna(subset=['user_id'])
        user_history_df = user_history_df.drop_duplicates(subset=['user_id'])

        user_sample = user_history_df['user_id'].sample(20)
        selected_user = st.selectbox("Select User", user_sample)

# ----- Main Panel -----
st.markdown("<h1 style='color:black;'>Musik Empfehlung:RhythMatch</h1>", unsafe_allow_html=True)

if rec_type == "Content-based":
    filtered_df = music_df if selected_artist == "All" else music_df[music_df['artist'] == selected_artist]
    song_list = sorted(filtered_df['name'].dropna().unique())

    song_name = st.selectbox("Select a song to get recommendations:", options=song_list)

    if song_name:
        recs = recommend_songs(song_name, music_df, numeric_features, text_features, tfidf, weight=sim_weight)

        st.subheader("Recommended Songs")
        for rec in recs:
            st.markdown(f"**{rec['name']}** by *{rec['artist']}*")
            st.caption(f"{rec['reason']}")
            if rec['url']:
                st.audio(rec['url'])
            st.markdown("---")

else:  # User-based recommendation
    if selected_user:
        top_songs = get_user_top_songs(user_history_df, music_df, selected_user)
        st.write(f"Top songs for user {selected_user}: {top_songs}")

        missing_songs = [song for song in top_songs if song not in music_df['name'].values]
        if missing_songs:
            st.write(f"Missing songs in music database: {missing_songs}")

        recs = recommend_songs_for_user(
            user_top_songs=top_songs,
            music_df=music_df,
            numeric_features=numeric_features,
            text_features=text_features,
            tfidf=tfidf,
            recommend_fn=recommend_songs,
            weight=sim_weight  # Now it's always defined
        )

        st.subheader(f"Recommendations for User {selected_user}")
        for rec in recs:
            st.markdown(f"**{rec['name']}** by *{rec['artist']}*")
            st.caption(f"{rec['reason']}")
            if rec['url']:
                st.audio(rec['url'])
            st.markdown("---")
