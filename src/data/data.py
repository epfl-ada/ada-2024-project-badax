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
    #more readable format
    metadata["Movie_genres"] = metadata["Movie_genres"].apply(lambda x: list(json.loads(x).values()))
    metadata.loc[metadata['Wikipedia_movie_ID']==29666067,'Movie_release_date'] = '2010-12-02'
    metadata["Movie_languages"] = metadata["Movie_languages"].apply(lambda x: list(json.loads(x).values()))
    metadata["Movie_countries"] = metadata["Movie_countries"].apply(lambda x: list(json.loads(x).values()))
    #fix country mappings
    mapping = {
        'Hong Kong': 'China',
        'West Germany': 'Germany',
        'Soviet Union': 'Russia',
        'Czechoslovakia': 'Czechia',
        'German Democratic Republic': 'Germany',
        'Yugoslavia': 'Serbia',
        'England': 'United Kingdom',
        'Weimar Republic': 'Germany',
        'Scotland': 'United Kingdom',
        'Korea': 'South Korea',
        'Burma': 'Myanmar',
        'Nazi Germany': 'Germany',
        'Republic of Macedonia': 'North Macedonia',
        'Socialist Federal Republic of Yugoslavia': 'Serbia',
        'Serbia and Montenegro': 'Serbia',
        'Kingdom of Great Britain': 'United Kingdom',
        'Federal Republic of Yugoslavia': 'Serbia',
        'Georgian SSR': 'Georgia',
        'Palestinian territories': 'Palestine',
        'Slovak Republic': 'Slovakia',
        'Mandatory Palestine': 'Palestine',
        'Uzbek SSR': 'Uzbekistan',
        'Wales': 'United Kingdom',
        'Northern Ireland': 'United Kingdom',
        'Ukranian SSR': 'Ukraine',
        'Isle of Man': 'United Kingdom',
        'Soviet occupation zone': 'Germany',
        'Malayalam Language': 'India',  # Language, assuming tied to India
        'Crime': 'Ukraine',  # Not clear, omitted
        'Iraqi Kurdistan': 'Iraq',
        'German Language': 'Germany',  # Language, assuming tied to Germany
        'Palestinian Territories': 'Palestine',
        'Kingdom of Italy': 'Italy',
        'Ukrainian SSR' : 'Ukraine',
        'Republic of China' : 'China',
        'Makau' : 'China',
        'Aruba' : 'Netherlands'
    }
    metadata["Movie_countries"] = metadata["Movie_countries"].apply(
        lambda countries: [mapping.get(country, country) for country in countries]
    )
    return metadata

def load_summaries():
    with open("data/MovieSummaries/plot_summaries.txt", "r", encoding="utf-8") as f:
        file = f.readlines()
    data = [line.strip().split("\t", 1) for line in file]
    summaries_df = pd.DataFrame(data, columns=["Wikipedia_movie_ID", "Movie_Summary"])
    summaries_df['Wikipedia_movie_ID'] = summaries_df['Wikipedia_movie_ID'].astype("int")
    
    return summaries_df




