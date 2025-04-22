import numpy as np

def get_user_top_songs(history_df, user_id, top_n=5):
    user_history = history_df[history_df['user_id'] == user_id]
    top_tracks = user_history.sort_values(by='playcount', ascending=False).head(top_n)
    return top_tracks['track_id'].tolist()

