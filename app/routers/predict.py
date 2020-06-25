import logging
import random

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, Field, validator

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import json

from joblib import load
model=load('knn_final.joblib')
df = pd.read_csv("https://raw.githubusercontent.com/BW-pilot/MachineLearning/master/spotify_final.csv")
spotify = df.drop(columns = ['track_id'])
scaler = StandardScaler()
spotify_scaled = scaler.fit_transform(spotify)

log = logging.getLogger(__name__)
router = APIRouter()

def knn_predictor(audio_feats, k=20):
    """
    differences_df = knn_predictor(audio_features)
    """
    audio_feats_scaled = scaler.transform([audio_feats])

    ##Nearest Neighbors model
    knn = model
    
    # make prediction 
    prediction = knn.kneighbors(audio_feats_scaled)

    # create an index for similar songs
    similar_songs_index = prediction[1][0][:k].tolist()

    # Create an empty list to store simlar song names
    similar_song_ids = []
    similar_song_names = []

    # loop over the indexes and append song names to empty list above
    for i in similar_songs_index:
        song_id = df['track_id'].iloc[i]
        similar_song_ids.append(song_id)

    #################################################

    column_names = spotify.columns.tolist()

    # put scaled audio features into a dataframe
    audio_feats_scaled_df = pd.DataFrame(audio_feats_scaled, columns=column_names)

    # create empty list of similar songs' features
    similar_songs_features = []

    # loop through the indexes of similar songs to get audio features for each
    #. similar song
    for index in similar_songs_index:
        list_of_feats = spotify.iloc[index].tolist()
        similar_songs_features.append(list_of_feats)

    # scale the features and turn them into a dataframe
    similar_feats_scaled = scaler.transform(similar_songs_features)
    similar_feats_scaled_df = pd.DataFrame(similar_feats_scaled, columns=column_names)

    # get the % difference between the outputs and input songs
    col_names = similar_feats_scaled_df.columns.to_list()
    diff_df = pd.DataFrame(columns=col_names)
    for i in range(k):
        diff = abs(similar_feats_scaled_df.iloc[i] - audio_feats_scaled_df.iloc[0])
        diff_df.loc[i] = diff

    # add sums of differences 
    diff_df['sum'] = diff_df.sum(axis=1)
    diff_df = diff_df.sort_values(by=['sum'])
    diff_df = diff_df.reset_index(drop=True)

    # add track_id to DF
    diff_df['track_id'] = similar_song_ids

    # reorder cols to have track_id as first column
    cols = list(diff_df)
    cols.insert(0, cols.pop(cols.index('track_id')))
    diff_df = diff_df.loc[:, cols]

    # Grab only the unique 10 songs
    diff_df = diff_df.drop_duplicates(subset=['track_id'])[:10]

    return diff_df


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    acousticness: float = Field(..., example=0.5)
    danceability: float = Field(..., example=0.7)
    energy: float = Field(..., example=0.5)
    liveness: float = Field(..., example=0.1)
    loudness: float = Field(..., example=-11.8)
    tempo: float = Field(..., example=-98.2)
    valence: float = Field(..., example=-0.6)
    instrumentalness: float = Field(..., example=-0.9)
    

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        dataframe = pd.DataFrame([dict(self)])
        return dataframe

@router.post('/predict')
async def predict(item: Item):    
    """Use a KNN model to made song predictions"""
    X_new = list(item)
    audio_features = []
    for i in X_new:
        audio_features.append(i[1])
    diff_df = knn_predictor(audio_features)
    something = diff_df.to_dict(orient='records')
    return JSONResponse(content=something)