##here we put all our functions so that the notebook is not messy
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from collections import Counter
from SPARQLWrapper import SPARQLWrapper, JSON

import time
import random


#does the query to match tconst and freebase ID - for mapping
def fetch_movies_with_ids(limit=5000):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    data = []
    offset = 0
    max_retries = 5  

    while True:
        # SPARQL query with LIMIT and OFFSET for pagination
        sparql.setQuery(f"""
            SELECT ?imdbID ?freebaseID WHERE {{
              ?movie wdt:P31 wd:Q11424.                # The movie is a film
              ?movie wdt:P345 ?imdbID.                 # The movie has an IMDb ID
              ?movie wdt:P646 ?freebaseID.             # The movie has a Freebase ID
            }}
            LIMIT {limit} OFFSET {offset}
        """)
        sparql.setReturnFormat(JSON)

        retries = 0

        while retries <= max_retries:
            try:
                # Execute query
                results = sparql.query().convert()

                # Check if there are results
                if not results["results"]["bindings"]:
                    return data  # Stop if no more results

                # Parse results
                for result in results["results"]["bindings"]:
                    imdb_id = result["imdbID"]["value"]
                    freebase_id = result["freebaseID"]["value"]
                    data.append([imdb_id, freebase_id])

                # Increment offset for next batch
                offset += limit
                break  # Exit retry loop if successful

            except Exception as e:
                retries += 1
                if retries > max_retries:
                    print(f"Max retries exceeded. Error: {e}")
                    return data

                # Handle "Too Many Requests" or other errors with exponential backoff
                wait_time = (2 ** retries) + random.uniform(0, 1)
                print(f"An error occurred: {e}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

    return data
    
#Gives list of countries from top1_df if saved in column of name movie_countries
def list_countries(top1_df, movie_countries):
    all_countries = chain.from_iterable(top1_df[movie_countries])
    
    country_counts = Counter(all_countries)
    
    country_counts_dict = dict(sorted(country_counts.items(), key=lambda item: item[1], reverse=True))
    df_country_counts = pd.DataFrame(list(country_counts_dict.items()), columns=['SOVEREIGNT', 'Occurrences'])
    return df_country_counts['SOVEREIGNT'].tolist()

#Given country and df with revenues, gives male and female ratio of revenues
def country_rev_ratio(country, top1_df):
    df = top1_df[top1_df["Movie_countries"].apply(lambda x: country in x if isinstance(x, list) else False)]
    aggregated_df = df.groupby('first_role_gender')['Movie_box_office_revenue'].sum().reset_index()
    try:
        m = aggregated_df.loc[aggregated_df["first_role_gender"] == "M", "Movie_box_office_revenue"].values[0]
    except:
        m = 0
    try:     
        f = aggregated_df.loc[aggregated_df["first_role_gender"] == "F", "Movie_box_office_revenue"].values[0]
    except:
        f = 0
    # print("m", m, "f", f)
    return {"M" : m/(m+f), "F": f/(m+f)}

#role importance based on idbm 
def assign_role_importance(merged_df):
    merged_df['first_role_actor'] = None
    merged_df['second_role_actor'] = None
    merged_df['third_role_actor'] = None
    merged_df['first_role_gender'] = None
    merged_df['second_role_gender'] = None
    merged_df['third_role_gender'] = None
    
    for index, row in merged_df.iterrows():
        actors = row['actors']
        genders = row['gender']
        
        if len(actors) > 0:
            merged_df.at[index, 'first_role_actor'] = actors[0]
            merged_df.at[index, 'first_role_gender'] = genders[0]
        if len(actors) > 1:
            merged_df.at[index, 'second_role_actor'] = actors[1]
            merged_df.at[index, 'second_role_gender'] = genders[1]
        if len(actors) > 2:
            merged_df.at[index, 'third_role_actor'] = actors[2]
            merged_df.at[index, 'third_role_gender'] = genders[2]
    
    return merged_df


#generation of metadata data frame
def metadata_gen():
    CMU_PATH = "MovieSummaries/MovieSummaries"
    metadata = pd.read_table(f"{CMU_PATH}/movie.metadata.tsv", header=None)
    
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
        'Aruba' : 'Netherlands',
        'United States of America' : 'United States'
    }
    metadata["Movie_genres"] = metadata["Movie_genres"].apply(lambda x: list(json.loads(x).values()))
    metadata["Movie_languages"] = metadata["Movie_languages"].apply(lambda x: list(json.loads(x).values()))
    metadata["Movie_countries"] = metadata["Movie_countries"].apply(lambda x: list(json.loads(x).values()))
    metadata["Movie_countries"] = metadata["Movie_countries"].apply(
        lambda countries: [mapping.get(country, country) for country in countries]
    )

    metadata['Movie_release_date'] = pd.to_numeric(metadata['Movie_release_date'].str[:4], errors='coerce')
    metadata['Movie_release_date'] = metadata['Movie_release_date'].fillna(0).astype(int)
    metadata.loc[metadata['Movie_release_date']==1010,'Movie_release_date'] = 2010
    metadata
    return metadata

#generation of characters data frame
def characters_gen():
    CMU_PATH = "MovieSummaries/MovieSummaries"
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

    characters =  characters[["Wikipedia movie ID", "Actor gender", "Actor name", "Actor age at movie release"]]

    characters = characters.groupby("Wikipedia movie ID").agg({
        "Actor gender": list, 
        "Actor name": list,
        "Actor age at movie release": list,
    }).reset_index()

    return characters

#generation of metadata_CMU data frame
def metadata_CMU_gen(metadata = None, characters = None):
    try: 
        if characters == None:
            characters = characters_gen()
    except:
        pass
    try:
        if metadata == None:
            metadata = metadata_gen()
    except:
        pass
    metadata_CMU = pd.merge(metadata, characters, on="Wikipedia movie ID", how="inner")
    return metadata_CMU

#generation of second_merge_df data frame 
def second_merge_df_gen(metadata = None):
    try:
        if metadata == None:
            metadata = metadata_gen()
    except:
        pass
    IMDB_PATH = "MovieSummaries/imdb_data"
    imdb_titles = pd.read_csv(f"{IMDB_PATH}/title.basics.tsv", sep="\t")
    imdb_titles = imdb_titles[imdb_titles["titleType"] == 'movie']
    imdb_titles = imdb_titles.drop(columns=["endYear"])
    imdb_titles["startYear"] = pd.to_numeric(imdb_titles["startYear"], errors="coerce")
    imdb_titles["startYear"] = imdb_titles["startYear"].fillna(0).astype(int)
    imdb_titles = imdb_titles.rename(columns = {"startYear":"Movie_release_date"})

    

    imdb_actors = pd.read_csv(f"{IMDB_PATH}/title.principals.tsv", sep="\t", engine = "pyarrow")
    imdb_actors = imdb_actors.drop(
        imdb_actors[~imdb_actors["category"].isin(["actor", "actress"])].index
    )
    imdb_actors = imdb_actors.drop_duplicates(subset=['tconst', 'nconst'])
    imdb_actors = imdb_actors[imdb_actors['tconst'].isin(imdb_titles['tconst'])]





    imdb_actors_names = pd.read_csv(f"{IMDB_PATH}/name.basics.tsv", sep="\t", engine = "pyarrow")



    imdb_actors = imdb_actors.merge(imdb_actors_names[['nconst', 'primaryName']], on='nconst', how='left')
    imdb_actors.category = imdb_actors.category.map({"actor": "M", "actress": "F"})
    imdb_actors = imdb_actors.rename(columns={"category":"gender"})
    imdb_actors_clean = imdb_actors.drop(columns = ["ordering", "job", "characters"])

    aggregate_actors = imdb_actors_clean.groupby("tconst").agg({
        "gender": list, 
        "primaryName": list,
    }).reset_index()

    imdb_titles = imdb_titles.merge(aggregate_actors, on = "tconst", how = "inner")
    imdb_titles = imdb_titles.rename(columns={"primaryName":"actors"})

    imdb_titles['len_match_imdb'] = imdb_titles.apply(
        lambda row: len(row['actors']) == len(row['gender']), axis=1
    )
    imdb_titles.drop(['titleType', 'primaryTitle', 'originalTitle', 'len_match_imdb', 'runtimeMinutes'], axis=1, inplace=True)

    
    data = fetch_movies_with_ids()
    tconst_freebase_df = pd.DataFrame(data, columns=["tconst", "Freebase_movie_ID"])

    first_merge_df = pd.merge(metadata, tconst_freebase_df, on=["Freebase_movie_ID"], how='inner')
    second_merge_df = pd.merge(first_merge_df, imdb_titles, on=["tconst"], how='inner')

    return second_merge_df

#generation of copy_merged_df_gen data frame
def copy_merged_df_gen(second_merge_df = None):
    try: 
        if second_merge_df == None:
            second_merge_df = second_merge_df_gen()
    except:
        pass
    return assign_role_importance(second_merge_df)

#function which helps efficiently create all needed file 
def all_data_gen():
    print("Starting all_data_gen...")
    metadata = metadata_gen()
    print("Starting all_data_gen...")
    characters = characters_gen()
    print("Starting all_data_gen...")
    metadata_CMU = metadata_CMU_gen(metadata, characters)
    print("Starting all_data_gen...")
    second_merge_df = second_merge_df_gen(metadata)
    print("Starting all_data_gen...")
    copy_merged_df = copy_merged_df_gen(second_merge_df)
    print("Starting all_data_gen...")
    return metadata_CMU, second_merge_df, copy_merged_df