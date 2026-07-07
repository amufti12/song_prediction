https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge

# Two-Stage Model for Automatic Playlist Continuation

Two-stage recommendation system that predicts playlist continuations using the Spotify Million Playlist Dataset. Built as a research project at Kennesaw State University with co-authors Raleigh Barden and Matthew Bowers.

## Overview

The goal was to predict the next song in a playlist given a sequence of preceding tracks. Rather than framing this as a direct classification problem, the model predicts a target audio feature vector, then retrieves the closest real songs to that vector. This two-stage design was inspired by Volkovs et al.'s first-place solution to the 2018 RecSys Challenge, simplified into two stages instead of their multi-model blended approach.

**Stage 1:** A recurrent neural network takes a variable-length sequence of a playlist's audio feature vectors and regresses to a predicted 13-dimensional feature vector representing the "next song."

**Stage 2:** A K-Nearest Neighbors model, trained on a track dataset of over 2 million songs, retrieves the closest real tracks to the predicted feature vector (k=500, matching the RecSys Challenge submission format).

This approach predicts what the user wants to hear in feature space first, then finds real songs that match, rather than restricting predictions to a fixed classification set. It also allows the model to accept variable-length playlist inputs.

## Technical Details

- **Data pipeline:** Collected audio features for all unique track URIs across the Million Playlist Dataset via the Spotify API (Spotipy), deduplicating songs with a set structure to minimize API calls. Cached the full track-feature dataset to a pickle file to avoid repeated API calls during training.
- **Input/output construction:** For each playlist, the last track was set as the prediction target (y), and the sequence of all preceding tracks became the input (x). Sequences were padded for batch processing.
- **Model architecture:** Sequential Keras model consisting of a masking layer (to ignore padding), stacked Bidirectional LSTM, Bidirectional GRU, and Bidirectional SimpleRNN layers with dropout, followed by a dense output layer producing the 13-feature prediction vector. Trained with the Adam optimizer, evaluated on MSE, RMSE, and MAE.
- **Retrieval model:** KNN trained on the full 2M+ song feature dataset, saved to disk to avoid retraining per playlist.
- **Baseline comparison:** A Random Forest model using the average feature vector of a playlist's tracks was tested as a simpler baseline but was dropped, since averaging features lost too much signal compared to the RNN's sequential prediction.
- **Tools:** Python, TensorFlow/Keras, scikit-learn, pandas, Spotipy.

## Key Findings

- The RNN showed a clear downward learning curve in loss and error metrics (MAE, RMSE) over 100 training epochs on a ~20,000 playlist sample (2% of the full dataset).
- Error decreased more rapidly when trained on a larger portion of the data, suggesting the model would benefit further from full-dataset training.
- Full-dataset training was constrained by compute: each epoch took approximately 24 hours at scale, limiting the project to proof-of-concept validation rather than full RecSys Challenge submission.
- Spotify's evaluation setup only provides scoring through official challenge submission (no held-out validation metrics), which limited the ability to benchmark final retrieval performance directly.

## Future Work

- Full-dataset training of the RNN stage.
- Incorporating an NLP model on playlist titles/descriptions to handle cold-start playlists (empty or title-only playlists), which were out of scope for this project.
- Exploring alternative distance/clustering methods for the retrieval stage beyond KNN.

## Reference

Built on the two-stage architecture proposed in: Volkovs, M., Rai, H., Cheng, Z., Wu, G., Lu, Y., & Sanner, S. (2018). *Two-Stage Model for Automatic Playlist Continuation at Scale.* RecSys Challenge '18.
