# import libraries
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from scipy.spatial.distance import mahalanobis

# load spotify dataset
data_dir = "C:/Users/whatk/SpotifyRecs/data"
df = pd.read_csv(f'{data_dir}/SpotifyTracks.csv')

# select features and preprocess data
features = ['danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
df_features = df[features]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)

# define autoencoder
input_dim = df_scaled.shape[1]  # Number of features
encoding_dim = 5  # Latent space dimensionality

# build autoencoder model
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = tf.keras.models.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# train autoencoder
autoencoder.fit(df_scaled, df_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# extract encoder part
encoder = tf.keras.models.Model(input_layer, encoded)

# encode all song features 
df_encoded = encoder.predict(df_scaled)

# compute covariance matrices 
cov_matrix = np.cov(df_encoded, rowvar=False)
inv_cov_matrix = np.linalg.inv(cov_matrix)

# create recommendation function using mahalanobis distance 
def recommend_songs(track_name, n_recommendations):
    # return error message if not valid song input
    if track_name not in df['track_name'].values:
        messagebox.showerror("Error", "Song not found in the dataset")
        return []
    
    # get index of input song
    track_index = df[df['track_name'] == track_name].index[0]
    
    # get latent representation of input song
    track_latent = df_encoded[track_index]
    
    # compute mahalanobis distance between the input song's latent vector and all others
    distances = [mahalanobis(track_latent, df_encoded[i], inv_cov_matrix) for i in range(len(df_encoded))]
    
    # get indices of closest songs based on mahalanobis distance
    closest_indices = np.argsort(distances)[1:]  # Skip the first one (itself)
    
    # keep track of recommended songs to ensure distinct 
    recommended_songs = pd.DataFrame(columns=['track_name', 'artists'])
    seen_tracks = set()  # Use a set to track unique (track_name, artist) pairs
    
    for index in closest_indices:
        track = df.iloc[index]
        track_info = (track['track_name'], track['artists'])
        
        # only add distinct combinations
        if track_info not in seen_tracks:
            seen_tracks.add(track_info)
            # Create a DataFrame row and concatenate it
            track_df = pd.DataFrame({'track_name': [track['track_name']], 'artists': [track['artists']]})
            recommended_songs = pd.concat([recommended_songs, track_df], ignore_index=True)
        
        # stop once we have the given number of recommendations
        if len(recommended_songs) >= int(n_recommendations):
            break
    
    return recommended_songs[['track_name', 'artists']]

# create function to update recommendation list in gui
def get_recommendations():
    track_name = track_entry.get()
    recommendation_num = recommend_count_entry.get()
    recommendations = recommend_songs(track_name, recommendation_num)
    
    # clear previous recommendations
    recommendation_listbox.delete(0, tk.END)
    
    if len(recommendations) == 0:
        return
    
    for index, row in recommendations.iterrows():
        recommendation_listbox.insert(tk.END, f"{row['track_name']} by {row['artists']}")

# initialize tkinter gui window
root = tk.Tk()
root.title("Spotify Recommendation System")

# add ui components
tk.Label(root, text="Enter a song name:").pack(pady=10)
track_entry = tk.Entry(root, width=40)
track_entry.pack(pady=10)

tk.Label(root, text="Enter how many recommendations you wish to receive:").pack(pady=5)
recommend_count_entry = tk.Entry(root, width=40)
recommend_count_entry.pack(pady=5)

recommend_button = tk.Button(root, text="Get Recommendations", command=get_recommendations)
recommend_button.pack(pady=10)

# display recommendations
recommendation_listbox = tk.Listbox(root, width=60, height=10)
recommendation_listbox.pack(pady=10)

# run the tkinter event loop
root.mainloop()
