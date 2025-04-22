# recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
        return [{
            "name": "Not Found",
            "artist": "Unknown",
            "reason": "Song not found in database.",
            "url": ""
        }]

    song_numeric_vector = numeric_features.iloc[song_idx].values.reshape(1, -1)
    song_text_vector = tfidf.transform([music_df.loc[song_idx, 'text_features']])
    song_genre = music_df.loc[song_idx, 'genre']
    song_tags = music_df.loc[song_idx, 'tags']

    numeric_sim = cosine_similarity(song_numeric_vector, numeric_features)[0]
    text_sim = cosine_similarity(song_text_vector, text_features)[0]
    combined_sim = weight * numeric_sim + (1 - weight) * text_sim

    top_indices = combined_sim.argsort()[::-1]
    recommendations = []

    for idx in top_indices:
        if idx == song_idx:
            continue
        if len(recommendations) >= 5:
            break

        row = music_df.iloc[idx]

        # Match reason logic
        if numeric_sim[idx] > text_sim[idx]:
            reason = "Matched on audio features 🎚️"
        else:
            matched_genres = set(song_genre.split(',')).intersection(set(row['genre'].split(',')))
            matched_tags = set(song_tags.split(',')).intersection(set(row['tags'].split(',')))
            genre_reason = f"Genre: {', '.join(matched_genres)}" if matched_genres else ""
            tag_reason = f"Tags: {', '.join(matched_tags)}" if matched_tags else ""
            reason = f"Matched on genre/tags 🎵 — {genre_reason} {tag_reason}".strip()

        recommendations.append({
            "name": row['name'],
            "artist": row['artist'],
            "reason": reason.strip(),
            "url": row.get('spotify_preview_url', '')
        })

    return recommendations
