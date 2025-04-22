import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_song_features(df):
    """
    Preprocesses the dataset to extract both numeric and text features.

    Args:
        df (pd.DataFrame): Raw music dataset.

    Returns:
        df (pd.DataFrame): Modified DataFrame with 'text_features' column.
        numeric_features (pd.DataFrame): Scaled audio-based numeric features.
        text_features (sparse matrix): TF-IDF vectorized genre + tags.
        tfidf (TfidfVectorizer): Fitted TF-IDF vectorizer (for reuse).
    """
    df = df.copy()

    # Handle missing values
    df['genre'] = df['genre'].fillna('').astype(str)
    df['tags'] = df['tags'].fillna('').astype(str)

    # Combine text fields
    df['text_features'] = df['genre'] + ' ' + df['tags']

    # Numeric audio features
    numeric_features = df[[
        'danceability', 'energy', 'acousticness',
        'instrumentalness', 'valence', 'tempo'
    ]]

    # TF-IDF on genre + tags
    tfidf = TfidfVectorizer()
    text_features = tfidf.fit_transform(df['text_features'])

    return df, numeric_features, text_features, tfidf


def recommend_songs(song_name, music_df, numeric_features, text_features, tfidf, weight=0.5):
    """
    Recommends top 5 similar songs based on a blend of audio and genre/tag similarity.

    Args:
        song_name (str): Song to base recommendations on.
        music_df (pd.DataFrame): Music metadata.
        numeric_features (pd.DataFrame): Audio features.
        text_features (sparse matrix): TF-IDF matrix.
        tfidf (TfidfVectorizer): Fitted vectorizer.
        weight (float): Weight for numeric features (0 = only text, 1 = only audio).

    Returns:
        List of recommendation dictionaries with song name, artist, reason, and preview URL.
    """
    try:
        song_idx = music_df[music_df['name'] == song_name].index[0]
    except IndexError:
        return [{
            "name": "Not Found",
            "artist": "Unknown",
            "reason": "Song not found in database.",
            "url": ""
        }]

    # Feature vectors for selected song
    song_numeric = numeric_features.iloc[song_idx]
    song_text_vector = tfidf.transform([music_df.loc[song_idx, 'text_features']])

    song_genre = music_df.loc[song_idx, 'genre']
    song_tags = music_df.loc[song_idx, 'tags']

    # Similarities
    numeric_sim = cosine_similarity([song_numeric], numeric_features)[0]
    text_sim = cosine_similarity(song_text_vector, text_features)[0]
    combined_sim = weight * numeric_sim + (1 - weight) * text_sim

    # Get top matches excluding the song itself
    top_indices = combined_sim.argsort()[::-1]
    recommendations = []

    for idx in top_indices:
        if idx == song_idx:
            continue
        if len(recommendations) >= 5:
            break

        row = music_df.iloc[idx]

        # Determine better match type
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
            "reason": reason,
            "url": row.get('spotify_preview_url', '')
        })

    return recommendations
