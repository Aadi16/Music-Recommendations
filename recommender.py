import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from explain import generate_reason

def create_song_features(df):
    # Copy the DataFrame to avoid mutating the original
    df = df.copy()

    # Fill missing values and ensure columns are string type
    df['genre'] = df['genre'].fillna('').astype(str)
    df['tags'] = df['tags'].fillna('').astype(str)

    # Create a combined text column for TF-IDF
    df['text_features'] = df['genre'] + ' ' + df['tags']

    # Numeric features for recommendation
    numeric_features = df[[
        'danceability', 'energy', 'acousticness',
        'instrumentalness', 'valence', 'tempo'
    ]]

    # Text vectorization using TF-IDF
    tfidf = TfidfVectorizer()
    text_features = tfidf.fit_transform(df['text_features'])

    # ✅ Return the modified DataFrame too
    return df, numeric_features, text_features, tfidf



def recommend_songs(song_name, music_df, numeric_features, text_features, tfidf, weight=0.5):
    try:
        # Find the index of the selected song
        song_idx = music_df[music_df['name'] == song_name].index[0]
    except IndexError:
        return [{"name": "Not Found", "artist": "Unknown", "reason": "Song not found in database.", "url": ""}]

    # Extract numeric features of the selected song
    song_numeric_features = numeric_features.iloc[song_idx]

    # Extract text features using TF-IDF for the selected song
    song_text = music_df.loc[song_idx, 'text_features']
    song_text_features = tfidf.transform([song_text])

    # Get the genres and tags of the selected song
    song_genre = music_df.loc[song_idx, 'genre']
    song_tags = music_df.loc[song_idx, 'tags']

    # Calculate cosine similarities
    numeric_sim = cosine_similarity([song_numeric_features], numeric_features)[0]
    text_sim = cosine_similarity(song_text_features, text_features)[0]

    # Combine both similarities (simple average, you can tune this)
    combined_sim = (numeric_sim * weight + text_sim * (1 - weight))

    # Get indices of top 5 most similar songs (excluding itself)
    top_indices = combined_sim.argsort()[::-1]

    recommendations = []
    for idx in top_indices:
        if idx != song_idx and len(recommendations) < 5:
            row = music_df.iloc[idx]
            reason = ""
            
            # If the match was mainly based on audio features
            if numeric_sim[idx] > text_sim[idx]:
                reason = f"Matched on audio features 🎚️"
            else:
                # If the match was based on genre/tags, add detailed reasoning
                matched_genres = set(song_genre.split(',')).intersection(set(row['genre'].split(',')))
                matched_tags = set(song_tags.split(',')).intersection(set(row['tags'].split(',')))

                genre_reason = f"Genre: {', '.join(matched_genres)}" if matched_genres else ""
                tag_reason = f"Tags: {', '.join(matched_tags)}" if matched_tags else ""
                
                reason = f"Matched on genre/tags 🎵 — {genre_reason} {tag_reason}"

            recommendations.append({
                "name": row['name'],
                "artist": row['artist'],
                "reason": reason.strip(),
                "url": row.get('spotify_preview_url', '')  # Assuming 'spotify_preview_url' contains the audio preview URL
            })

    return recommendations
