import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_song_features(df):
    df = df.copy()
    df['genre'] = df['genre'].fillna('').astype(str)
    df['tags'] = df['tags'].fillna('').astype(str)
    df['text_features'] = df['genre'] + ' ' + df['tags']

    numeric_cols = ['danceability', 'energy', 'acousticness',
                    'instrumentalness', 'valence', 'tempo']
    numeric_features = df[numeric_cols]

    tfidf = TfidfVectorizer()
    text_features = tfidf.fit_transform(df['text_features'])

    return df, numeric_features, text_features, tfidf


def recommend_songs(song_name, music_df, numeric_features, text_features, tfidf, weight=0.5, top_n=5):
    try:
        song_idx = music_df[music_df['name'] == song_name].index[0]
    except IndexError:
        return [{
            "name": "Not Found",
            "artist": "Unknown",
            "reason": "Song not found in database.",
            "url": ""
        }]

    song_numeric = numeric_features.iloc[song_idx].values.reshape(1, -1)
    song_text = tfidf.transform([music_df.loc[song_idx, 'text_features']])

    sim_audio = cosine_similarity(song_numeric, numeric_features)[0]
    sim_text = cosine_similarity(song_text, text_features)[0]

    music_df = music_df.copy()
    music_df['sim_audio'] = sim_audio
    music_df['sim_text'] = sim_text

    # Exclude the input song itself
    music_df = music_df.drop(index=song_idx)

    # Parse tags/genres for reasoning
    orig_genres = set(music_df.iloc[song_idx]['genre'].split(','))
    orig_tags = set(music_df.iloc[song_idx]['tags'].split(','))

    if weight < 0.5:
        mode = 'text'
        music_df['score'] = music_df['sim_text']
    elif weight > 0.5:
        mode = 'audio'
        music_df['score'] = music_df['sim_audio']
    else:
        mode = 'hybrid'
        music_df['score'] = (music_df['sim_audio'] + music_df['sim_text']) / 2

    # Add diversity boost using tag/genre overlap
    def overlap_boost(row):
        genres = set(row['genre'].split(','))
        tags = set(row['tags'].split(','))
        genre_overlap = len(orig_genres & genres)
        tag_overlap = len(orig_tags & tags)
        return genre_overlap + tag_overlap

    music_df['diversity_score'] = music_df.apply(overlap_boost, axis=1)

    # Hybrid score to promote diverse but relevant results
    if mode == 'text':
        music_df['final_score'] = music_df['score'] + 0.1 * music_df['diversity_score']
    elif mode == 'audio':
        music_df['final_score'] = music_df['score'] + 0.05 * np.random.rand(len(music_df))  # Slight randomness
    else:  # hybrid
        music_df['final_score'] = (
            0.5 * music_df['sim_audio'] +
            0.5 * music_df['sim_text'] +
            0.05 * music_df['diversity_score']
        )

    music_df = music_df.sort_values(by='final_score', ascending=False).head(30).sample(n=top_n, random_state=None)

    # Final result assembly
    recommendations = []
    for _, row in music_df.iterrows():
        rec_genre = set(row['genre'].split(','))
        rec_tags = set(row['tags'].split(','))
        matched_genres = orig_genres & rec_genre
        matched_tags = orig_tags & rec_tags

        if mode == "text":
            reason = f"Recommended based on tags/genre — Genre: {', '.join(matched_genres) or 'N/A'}, Tags: {', '.join(matched_tags) or 'N/A'}"
        elif mode == "audio":
            reason = f"Recommended based on audio features"
        else:
            reason = f"Balanced recommendation — Genre: {', '.join(matched_genres) or 'N/A'}, Tags: {', '.join(matched_tags) or 'N/A'}"

        recommendations.append({
            "name": row['name'],
            "artist": row['artist'],
            "reason": reason,
            "url": row.get('spotify_preview_url', '')
        })

    return recommendations
