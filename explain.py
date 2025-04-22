def generate_reason(selected_song_row, recommended_song_row, numeric_sim, text_sim, idx):
    # Compare numeric features individually (for optional expansion)
    feature_comparisons = []

    feature_names = ['danceability', 'energy', 'acousticness', 'instrumentalness', 'valence', 'tempo']
    for feature in feature_names:
        selected_value = selected_song_row[feature]
        recommended_value = recommended_song_row[feature]
        diff = abs(selected_value - recommended_value)
        if diff < 0.1:  # You can adjust threshold for "similar"
            feature_comparisons.append(f"{feature} ({recommended_value:.2f})")

    if numeric_sim > text_sim:
        reason = f"Audio Match — Similar {', '.join(feature_comparisons)}"
    else:
        genre = recommended_song_row.get('genre', 'Unknown')
        tags = recommended_song_row.get('tags', 'None')
        reason = f"Genre/Tag Match — Genre: {genre}, Tags: {tags}"

    return reason
