import streamlit as st
import pandas as pd
from recommender import create_song_features, recommend_songs

# Load your music dataset
music_df = pd.read_csv("music_df_cleaned.csv")  # Change path if needed

# Create features
music_df, numeric_features, text_features, tfidf = create_song_features(music_df)

# Streamlit UI

# Customizing the sidebar
st.sidebar.title("🎧 Filters & Settings")

# --- Filter by artist (optional) ---
with st.sidebar.expander("🎤 Filter by Artist"):
    artist_list = sorted(music_df['artist'].dropna().unique())
    selected_artist = st.sidebar.selectbox("Filter songs by artist:", ["All"] + artist_list)

# --- Similarity weighting slider ---
with st.sidebar.expander("⚖️ Similarity Preference: Audio vs. Genre/Tags"):
    sim_weight = st.sidebar.slider(
        "Adjust similarity preference",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0.0 = Only text (genre/tags), 1.0 = Only audio features"
    )

# Main Panel

# Title and description
st.title("🎧 Smart Music Recommender")
st.markdown("Welcome to the Smart Music Recommender! 🎶 Select a song to get similar song recommendations based on audio features and genre/tags.")

# --- Filtered song list ---
if selected_artist != "All":
    filtered_df = music_df[music_df['artist'] == selected_artist]
else:
    filtered_df = music_df

song_list = filtered_df['name'].dropna().unique()

# Song search dropdown
song_name = st.selectbox("🎵 Select a song to get recommendations:", options=sorted(song_list))

# Generate recommendations
if song_name:
    recs = recommend_songs(
        song_name,
        music_df,
        numeric_features,
        text_features,
        tfidf,
        weight=sim_weight  # 👈 added new param
    )

    # Display the recommendations
    st.subheader("🎯 Recommended Songs:")
    if recs and recs[0]["name"] == "Not Found":
        st.warning("Song not found in the database.")
    else:
        for rec in recs:
            st.markdown(f"**{rec['name']}** by *{rec['artist']}*")
            st.markdown(f"_Reason: {rec['reason']}_")
            
            # Check if there's a URL for the audio preview
            if rec.get("url"):  
                st.audio(rec["url"])  # Display audio preview

            st.markdown("---")

