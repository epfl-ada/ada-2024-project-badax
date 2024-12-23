#Functions to load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

CMU_PATH = "data/MovieSummaries"
IMDB_PATH = "data/imdb_data"

def load_metadata():
    metadata = pd.read_table("data/MovieSummaries/movie.metadata.tsv", header=None)
    metadata.columns = [
        "Wikipedia movie ID",
        "Freebase_movie_ID",
        "Movie_name",
        "Movie_release_date",
        "Movie_box_office_revenue",
        "Movie_runtime",
        "Movie_languages", 
        "Movie_countries",  
        "Movie_genres"     
    ]
    #more readable format
    metadata["Movie_genres"] = metadata["Movie_genres"].apply(lambda x: list(json.loads(x).values()))
    # Only keep year in date
    metadata['Movie_release_date'] = pd.to_numeric(metadata['Movie_release_date'].str[:4], errors='coerce')
    metadata['Movie_release_date'] = metadata['Movie_release_date'].fillna(0).astype(int)
    metadata.loc[metadata['Wikipedia movie ID']==29666067,'Movie_release_date'] = 2010
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

def load_characters():
    characters = pd.read_table(f"{CMU_PATH}/character.metadata.tsv", header=None)

    characters.columns = [
    "Wikipedia movie ID",
    "Freebase movie ID",
    "Movie release date",
    "Character name",
    "Actor date of birth",
    "Actor gender",
    "Actor height (in meters)",
    "Actor ethnicity (Freebase ID)",
    "Actor name",
    "Actor age at movie release",
    "Freebase character/actor map ID",
    "Freebase character ID",
    "Freebase actor ID"
    ]
    return characters

def load_summaries():
    with open("data/MovieSummaries/plot_summaries.txt", "r", encoding="utf-8") as f:
        file = f.readlines()
    data = [line.strip().split("\t", 1) for line in file]
    summaries_df = pd.DataFrame(data, columns=["Wikipedia movie ID", "Movie_Summary"])
    summaries_df['Wikipedia movie ID'] = summaries_df['Wikipedia movie ID'].astype("int")
    
    return summaries_df

def load_imdb_titles():
    imdb_titles = pd.read_csv(f"{IMDB_PATH}/title.basics.tsv", sep="\t")
    print('We drop all the rows that do not describe movies')
    imdb_titles = imdb_titles[imdb_titles["titleType"] == 'movie']
    print('We verify we have only one title type:')
    print(imdb_titles.nunique())
    print('We can see that endYear is not really useful (only 1 value). We decide to drop it and only use startYear as reference for later.')
    imdb_titles = imdb_titles.drop(columns=["endYear"])
    print('We convert start year into integers so we can easily plot them afterwards. We also convert NaNs to 0 so that we can avoid errors later.')
    imdb_titles["startYear"] = pd.to_numeric(imdb_titles["startYear"], errors="coerce")
    imdb_titles["startYear"] = imdb_titles["startYear"].fillna(0).astype(int)
    imdb_titles = imdb_titles.rename(columns = {"startYear":"Movie_release_date"})
    return imdb_titles

def load_imdb_actors(imdb_titles):
    imdb_actors = pd.read_csv(f"{IMDB_PATH}/title.principals.tsv", sep="\t", engine = "pyarrow")
    print('We drop all movie staff that is not an actor or actress')
    imdb_actors = imdb_actors.drop(
        imdb_actors[~imdb_actors["category"].isin(["actor", "actress"])].index
    )
    print('We drop duplicate actors (that appear in the same movie more than once)')
    imdb_actors = imdb_actors.drop_duplicates(subset=['tconst', 'nconst'])
    print('We only want to keep actors which are in our movies dataset')
    imdb_actors = imdb_actors[imdb_actors['tconst'].isin(imdb_titles['tconst'])]
    print('Adding actor names')
    imdb_actors_names = pd.read_csv(f"{IMDB_PATH}/name.basics.tsv", sep="\t", engine = "pyarrow")
    print('We assign actors ids with their other information by merging imdb_actors and imdb_actors_names')
    imdb_actors = imdb_actors.merge(imdb_actors_names[['nconst', 'primaryName']], on='nconst', how='left')
    print("Mapping the gender for readability, renaming the column's name and dropping unused columns")
    imdb_actors.category = imdb_actors.category.map({"actor": "M", "actress": "F"})
    imdb_actors = imdb_actors.rename(columns={"category":"gender"})
    imdb_actors_clean = imdb_actors.drop(columns = ["ordering", "job", "characters"])
    return imdb_actors_clean

def get_preliminary_dfs():
    CMU_PATH = "data/MovieSummaries/"
    IMDB_PATH = "data/imdb_data/"

    movie_data_unmerged = pd.read_table(CMU_PATH + "movie.metadata.tsv", header=None)

    movie_data_unmerged.columns = [
        "Wikipedia_movie_ID",
        "Freebase_movie_ID",
        "Movie_name",
        "Movie_release_date",
        "Movie_box_office_revenue",
        "Movie_runtime",
        "Movie_languages", 
        "Movie_countries",  
        "Movie_genres"     
    ]

    with open(CMU_PATH + "plot_summaries.txt", "r", encoding="utf-8") as f:
        file = f.readlines()

    data = [line.strip().split("\t", 1) for line in file]

    summaries_df = pd.DataFrame(data, columns=["Wikipedia_movie_ID", "Movie_Summary"])

    summaries_df['Wikipedia_movie_ID'] = summaries_df['Wikipedia_movie_ID'].apply(int)

    df_character = pd.read_table(CMU_PATH+"character.metadata.tsv", header=None)

    df_character.columns = ["Wikipedia_movie_ID",
    "Freebase_movie_ID",
    "Movie_release_date",
    "Character_name",
    "Actor_date_of_birth",
    "Actor_gender",
    "Actor_height", #in meters
    "Actor_ethnicity",
    "Actor_name",
    "Actor_age_at_movie_release",
    "Freebase_character/actor_map_ID",
    "Freebase_character_ID",
    "Freebase_actor_ID"]

    df_character["Wikipedia_movie_ID"] = df_character["Wikipedia_movie_ID"].apply(int)
    #here we only keep the year (and not the month) + we put the same format
    df_character["Movie_release_date"] = df_character["Movie_release_date"].apply(lambda x: int(str(x).split("-")[0]) if type(x) == str else np.nan)

    df_movie = movie_data_unmerged.merge(summaries_df, on="Wikipedia_movie_ID", how='inner')

    df_movie["Movie_genres"] = df_movie["Movie_genres"].apply(lambda x: list(json.loads(x).values()))
    #some movies have release date in the form year-month-day, we only want the year, and some have nan values
    df_movie["Movie_release_date"] = df_movie["Movie_release_date"].apply(lambda x: int(str(x).split("-")[0]) if type(x) == str else np.nan)
    df_movie.loc[df_movie["Movie_name"] == "Hunting Season", "Movie_release_date"] = 2010

    def find_movie_ID(movie_name):
        return int(df_movie[df_movie["Movie_name"] == movie_name]["Wikipedia_movie_ID"].values[0])

    df_imdb = pd.read_csv(IMDB_PATH +'merged_imdb_2.csv')
    df_imdb.head()
    df_imdb["actors"] = df_imdb["actors"].apply(lambda x: ast.literal_eval(x))

    df_imdb.rename(columns={"Wikipedia movie ID": "Wikipedia_movie_ID"}, inplace=True)

    df_actor = pd.DataFrame()
    df_actor["actor"] = [item for sublist in df_imdb['actors'].tolist() for item in sublist]
    df_actor = df_actor.drop_duplicates()
    df_movie = df_movie.merge(df_imdb[["gender", "Wikipedia_movie_ID"]], on="Wikipedia_movie_ID", how='inner')
    df_movie["Movie_countries"] = df_movie["Movie_countries"].apply(lambda x: list(json.loads(x).values()))
    return df_movie, df_character, df_actor