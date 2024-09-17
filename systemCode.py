import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# load spotify dataset
data_dir = "data/SpotifyTracks.csv"
df = pd.read_csv(data_dir)

# Select features and preprocess data (same steps as before)
features = ['danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
df_features = df[features]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)
similarity_matrix = cosine_similarity(df_scaled)

# Recommendation function
def recommend_songs(track_name, n_recommendations=10):
    if track_name not in df['track_name'].values:
        messagebox.showerror("Error", "Song not found in the dataset")
        return []

    track_index = df[df['track_name'] == track_name].index[0]
    sim_scores = list(enumerate(similarity_matrix[track_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recommendations + 1]  # Skip the first result as it's the same song

    song_indices = [i[0] for i in sim_scores]
    recommended_songs = df.iloc[song_indices]
    return recommended_songs[['track_name', 'artist_name']]

# Function to update the recommendation list in the GUI
def get_recommendations():
    track_name = track_entry.get()
    recommendations = recommend_songs(track_name)
    
    # Clear previous recommendations
    recommendation_listbox.delete(0, tk.END)
    
    if len(recommendations) == 0:
        return
    
    for index, row in recommendations.iterrows():
        recommendation_listbox.insert(tk.END, f"{row['track_name']} by {row['artist_name']}")

# Initialize the Tkinter GUI window
root = tk.Tk()
root.title("Spotify Recommendation System")

# Add UI components
tk.Label(root, text="Enter a song name:").pack(pady=10)
track_entry = tk.Entry(root, width=40)
track_entry.pack(pady=10)

recommend_button = tk.Button(root, text="Get Recommendations", command=get_recommendations)
recommend_button.pack(pady=10)

# Listbox to display recommendations
recommendation_listbox = tk.Listbox(root, width=60, height=10)
recommendation_listbox.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
