import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from explain import generate_reason
import numpy as np

def create_song_features(df):
    df = df.copy()
    df['genre'] = df['genre'].fillna('').astype(str)
    df['tags'] = df['tags'].fillna('').astype(str)
    df['text_features'] = df['genre'] + ' ' + df['tags']

    numeric_features = df[[
        'danceability', 'energy', 'acousticness',
        'instrumentalness', 'valence', 'tempo'
    ]]

    tfidf = TfidfVectorizer()
    text_features = tfidf.fit_transform(df['text_features'])

    return df, numeric_features, text_features, tfidf




def recommend_songs(song_name, music_df, numeric_features, text_features, tfidf, weight=0.5):
    try:
        song_idx = music_df[music_df['name'] == song_name].index[0]
    except IndexError:
        return [{"name": "Not Found", "artist": "Unknown", "reason": "Song not found in database.", "url": ""}]

    # Extract features for the selected song
    song_numeric = numeric_features.iloc[song_idx]
    song_text_raw = music_df.loc[song_idx, 'text_features']
    song_text_vector = tfidf.transform([song_text_raw])

    song_genre = music_df.loc[song_idx, 'genre']
    song_tags = music_df.loc[song_idx, 'tags']

    # Compute similarity
    numeric_sim = cosine_similarity([song_numeric], numeric_features)[0]
    text_sim = cosine_similarity(song_text_vector, text_features)[0]

    # Normalize both similarities
    numeric_sim_norm = (numeric_sim - np.min(numeric_sim)) / (np.max(numeric_sim) - np.min(numeric_sim))
    text_sim_norm = (text_sim - np.min(text_sim)) / (np.max(text_sim) - np.min(text_sim))

    # Combine based on weight
    combined_sim = weight * numeric_sim_norm + (1 - weight) * text_sim_norm

    # Sort by final score
    top_indices = np.argsort(combined_sim)[::-1]

    recommendations = []

    for idx in top_indices:
        if idx == song_idx:
            continue  # skip the selected song
        if len(recommendations) >= 5:
            break

        row = music_df.iloc[idx]

        # Generate reason
        if weight >= 0.65:
            reason = f"Matched on audio features"
        elif weight <= 0.35:
            matched_genres = set(song_genre.split(',')).intersection(set(row['genre'].split(',')))
            matched_tags = set(song_tags.split(',')).intersection(set(row['tags'].split(',')))
            genre_reason = f"Genre: {', '.join(matched_genres)}" if matched_genres else ""
            tag_reason = f"Tags: {', '.join(matched_tags)}" if matched_tags else ""
            reason = f"Matched on genre/tags — {genre_reason} {tag_reason}".strip()
        else:
            reason = "Balanced match — blend of audio and genre/tags"

        recommendations.append({
            "name": row['name'],
            "artist": row['artist'],
            "reason": reason,
            "url": row.get('spotify_preview_url', '')
        })

    return recommendations
