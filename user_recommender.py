# user_recommender.py
import pandas as pd

def get_user_top_songs(history_df, music_df, user_id, top_n=10):
    user_history = history_df[history_df['user_id'] == user_id]
    top_tracks = user_history.sort_values(by='playcount', ascending=False).head(top_n)

    # Map track_id to song name using music_df
    track_id_to_name = pd.Series(music_df['name'].values, index=music_df['track_id']).to_dict()
    top_song_names = [track_id_to_name.get(track_id) for track_id in top_tracks['track_id']]
    
    # Filter out None values if some track_ids weren't found
    return [name for name in top_song_names if name is not None]


def recommend_songs_for_user(user_top_songs, music_df, numeric_features, text_features, tfidf, recommend_fn, weight=0.5):
    all_recommendations = []

    for song in user_top_songs:
        recs = recommend_fn(song, music_df, numeric_features, text_features, tfidf, weight=weight)
        all_recommendations.extend(recs)

    # Optional: Deduplicate by song name
    seen = set()
    unique_recs = []
    for rec in all_recommendations:
        if rec['name'] not in seen:
            unique_recs.append(rec)
            seen.add(rec['name'])

    return unique_recs[:10]  # limit final recommendations
