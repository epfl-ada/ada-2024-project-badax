#Functions to load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

def load_metadata():
    metadata = pd.read_table("data/MovieSummaries/movie.metadata.tsv", header=None)
    metadata.columns = [
        "Wikipedia_movie_ID",
        "Freebase_movie_ID",
        "Movie_name",
        "Movie_release_date",
        "Movie_box_office_revenue",
        "Movie_runtime",
        "Movie_languages", 
        "Movie_countxries",  
        "Movie_genres"     
    ]
    metadata["Movie_genres"] = metadata["Movie_genres"].apply(lambda x: list(json.loads(x).values()))
    metadata.loc[metadata['Wikipedia_movie_ID']==29666067,'Movie_release_date'] = '2010-12-02'
    return metadata

def load_summaries():
    with open("data/MovieSummaries/plot_summaries.txt", "r", encoding="utf-8") as f:
        file = f.readlines()
    data = [line.strip().split("\t", 1) for line in file]
    summaries_df = pd.DataFrame(data, columns=["Wikipedia_movie_ID", "Movie_Summary"])
    summaries_df['Wikipedia_movie_ID'] = summaries_df['Wikipedia_movie_ID'].astype("int")
    return summaries_df




