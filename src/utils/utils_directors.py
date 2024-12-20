import numpy as np
import pandas as pd
import wikipediaapi
import requests
import ast
import time
import random
from urllib.error import HTTPError
import plotly.express as px
import gender_guesser.detector as gender
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Pool
from SPARQLWrapper import SPARQLWrapper, JSON


def batch_query_wikidata(imdb_ids, sparql):
    """
    Query Wikidata for directors given a batch of IMDb IDs.
    param imdb_ids: List of IMDb IDs to query
    param sparql: SPARQLWrapper instance
    """
    values_clause = " ".join(f'"{imdb_id}"' for imdb_id in imdb_ids)
    query = f"""
    SELECT ?imdbID ?director ?directorLabel WHERE {{
      VALUES ?imdbID {{{values_clause}}}
      ?movie wdt:P345 ?imdbID;           # IMDb ID property
             wdt:P57 ?director.          # Director property
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".  # English labels
      }}
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    directors_info = {}
    for result in results["results"]["bindings"]:
        imdb_id = result["imdbID"]["value"]
        director_name = result["directorLabel"]["value"]
        if imdb_id not in directors_info:
            directors_info[imdb_id] = []
        directors_info[imdb_id].append({"director": director_name})
    return directors_info

def process_batch_with_backoff(batch, sparql, retries=5):
    """
    Process a batch of IMDb IDs with exponential backoff in case of errors.
    param batch: List of IMDb IDs to process
    param retries: Number of retries before giving up
    """
    wait_time = 1  
    for attempt in range(retries):
        try:
            return batch_query_wikidata(batch, sparql)
        except Exception as e:
            print(f"Error: {e}. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})")
            time.sleep(wait_time)
            wait_time *= 2 
    print(f"Failed to process batch after {retries} retries.")
    return {}


def fetch_gender_wikidata(offset, sparql, chunk_size=100):
    """Fetch data from Wikidata with exponential backoff in case of errors.
    param offset: Offset for the query
    param sparql: SPARQLWrapper instance
    param chunk_size: Number of results to fetch
    
    """
    max_retries = 5  
    backoff_factor = 2  
    retry_delay = 2  

    for attempt in range(max_retries):
        try:
            sparql.setQuery(f"""
            SELECT DISTINCT
              ?filmDirector ?filmDirectorLabel ?gender ?genderLabel
            WHERE 
            {{
              ?filmDirector p:P106 ?statement1.
              ?statement1 (ps:P106/(wdt:P279*)) wd:Q2526255. # Find all items with a particular profession (director)
              OPTIONAL {{?filmDirector wdt:P21 ?gender.}}      
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
            }}
            LIMIT {chunk_size}
            OFFSET {offset}
            """)
            
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(120)
            results = sparql.query().convert()
            
            data = []
            for result in results["results"]["bindings"]:
                data.append({
                    "filmDirector": result.get("filmDirectorLabel", {}).get("value", None),
                    "gender": result.get("genderLabel", {}).get("value", None),
                })
            return data
        
        except HTTPError as e:
            if e.code == 429: 
                if attempt < max_retries - 1: 
                    delay = retry_delay * (backoff_factor ** attempt)
                    print(f"HTTP 429: Retrying in {delay:.2f} seconds...")
                    time.sleep(delay + random.uniform(0, 1))  
            elif e.code == 500:
                print(f"HTTP 500 at OFFSET {offset}: Retrying after delay...")
                time.sleep(10)  
            else:
                print("Max retries reached. Exiting.")
                raise


def get_gender_from_namsor(first_name, last_name, api_key):
    """Determine gender using NamSor API."""
    url = "https://v2.namsor.com/NamSorAPIv2/api2/json/genderBatch"
    headers = {
        "X-API-KEY": api_key,
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "personalNames": [
            {
                "id": "unique-id", 
                "firstName": first_name,
                "lastName": last_name
            }
        ]
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if 'personalNames' in data and len(data['personalNames']) > 0:
                person = data['personalNames'][0]
                return person.get('likelyGender'), 'NamSor'
    except Exception as e:
        print(f"Error fetching data from NamSor API: {e}")
    return np.nan, 'Unknown'


def get_gender_from_wikipedia(name, NAMSOR_API_KEY):
    """Determine gender using the French and English Wikipedia pages."""
    wiki_fr = wikipediaapi.Wikipedia("MyProjectName (merlin@example.com)", 'fr')
    wiki_en = wikipediaapi.Wikipedia("MyProjectName (merlin@example.com)", 'en')
    # We first try the french wikipedia
    page_fr = wiki_fr.page(name)
    if page_fr.exists():
        content_fr = page_fr.text.lower()
        if 'profession' in content_fr:
            if 'réalisateur' in content_fr or 'producteur' in content_fr:
                return 'male', 'Wikipedia French'
            elif 'réalisatrice' in content_fr or 'productrice' in content_fr:
                return 'female', 'Wikipedia French'
        elif 'réalisateur' in content_fr or 'réalisatrice' in content_fr:
            if 'réalisateur' in content_fr:
                return 'male', 'Wikipedia French'                         
            elif 'réalisatrice' in content_fr:
                return 'female', 'Wikipedia French'

    # If we don't succeed, we try english wikipedia
    page_en = wiki_en.page(name)
    if page_en.exists():
        content_en = page_en.text.lower()
        male_keywords = ['he', 'him', 'his', 'himself']
        female_keywords = ['she', 'her', 'hers', 'herself']
        male_count = sum(content_en.count(word) for word in male_keywords)
        female_count = sum(content_en.count(word) for word in female_keywords)
        if male_count > female_count:
            return 'male', 'Wikipedia English'
        elif female_count > male_count:
            return 'female', 'Wikipedia English'

    # If wikipedia does not work, use gender-guesser
    d = gender.Detector()
    guessed_gender = d.get_gender(name.split()[0])  # Use first name
    if guessed_gender in ['male', 'mostly_male']:
        return 'male', 'Gender Guesser'
    elif guessed_gender in ['female', 'mostly_female']:
        return 'female', 'Gender Guesser'

    # Else use namsor
    return get_gender_from_namsor(name.split()[0], name.split()[-1], NAMSOR_API_KEY)

def process_gender_row_dict(row, NAMSOR_API_KEY):
    """Process gender for a single row represented as a dictionary."""
    name, current_gender = row['director'], row['gender_directors']
    if pd.isna(current_gender):  # Check if the current gender is NaN
        gender, source = get_gender_from_wikipedia(name, NAMSOR_API_KEY)
        return {'Gender': gender, 'Gender Source': source}
    return {'Gender': current_gender, 'Gender Source': 'Existing'}

def process_gender_parallel(data,NAMSOR_API_KEY, n_cores=4):
    """Parallel processing using joblib."""

    rows = data.to_dict(orient='records')

    results = Parallel(n_jobs=n_cores)(
        delayed(process_gender_row_dict)(row, NAMSOR_API_KEY) for row in tqdm(rows, desc="Processing")
    )

    results_df = pd.DataFrame(results)
    return pd.concat([data.reset_index(drop=True), results_df], axis=1)