from sklearn.metrics.pairwise import cosine_similarity

def recommend_songs(user_song_ids, music_df, numeric_features, text_features, top_n=5):
    # Get indices of songs the user liked
    liked_indices = music_df[music_df['track_id'].isin(user_song_ids)].index

    # Average feature vector of liked songs
    user_profile_vector = numeric_features.iloc[liked_indices].mean().values.reshape(1, -1)

    # Compute similarity with all songs
    similarity_scores = cosine_similarity(user_profile_vector, numeric_features).flatten()

    # Get top N indices excluding already listened
    music_df['similarity'] = similarity_scores
    recommendations = music_df[~music_df['track_id'].isin(user_song_ids)] \
                        .sort_values(by='similarity', ascending=False).head(top_n)
    
    return recommendations
