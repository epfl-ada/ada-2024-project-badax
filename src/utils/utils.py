import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

try:
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
except OSError:
    print("Model 'en_core_web_sm' not found. Downloading...")
    spacy.cli.download("en_core_web_sm", disable=['parser', 'ner'])
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    



def assign_role_importance(merged_df):
    """
    Function to assign first, second, and third roles for actors and genders.
    
    Parameters:
    - merged_df: DataFrame containing the columns 'actors' and 'gender' (lists).
    
    Returns:
    - A DataFrame with columns 'first_role', 'second_role', 'third_role' for actors and genders.
    """
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

def country_female_ratio(merged_df, country):
    """
    Function to calculate and plot the female ratio among the top three roles of movies produced in a specified country over time.
    
    Parameters:
    - merged_df: DataFrame containing movie information
    - country: the desired country for which we willconductt the calculation
    
    Returns:
    - The standard deviation of the yearly female ratio (float).
    """
    
    merged_df = merged_df[merged_df["Movie_countries"].apply(lambda x: country in x and len(x) == 1)]
    
    merged_df.loc[:, 'F'] = (merged_df[['first_role_gender', 'second_role_gender', 'third_role_gender']] == 'F').sum(axis=1)
    merged_df.loc[:, 'M'] = (merged_df[['first_role_gender', 'second_role_gender', 'third_role_gender']] == 'M').sum(axis=1)
    gender_year_df = merged_df[merged_df["Movie_release_date_x"] > 0]
    gender_year_df = gender_year_df[["Movie_release_date_x", "F", "M"]]
    gender_year_df.loc[:,"sum"] = gender_year_df["F"] + gender_year_df["M"]
    gender_year_sum_df = gender_year_df.groupby("Movie_release_date_x").agg({
    "F": "sum", 
    "M": "sum",
    "sum": "sum",
    }).reset_index()
    gender_year_sum_df = gender_year_sum_df[gender_year_sum_df["sum"] > 0]
    gender_year_sum_df.loc[:, "Female_ratio"] = gender_year_sum_df["F"]/gender_year_sum_df["sum"]
    gender_year_sum_df = gender_year_sum_df[["Movie_release_date_x", "Female_ratio"]]
    standard_dev = gender_year_sum_df["Female_ratio"].std()
    plt.plot(gender_year_sum_df['Movie_release_date_x'], gender_year_sum_df['Female_ratio'], color='teal', marker='o', linestyle='-', linewidth=2)
    #plt.text(1965, 0, f"Standard deviation: {standard_dev*100:.2f}%", fontsize=12, color='red')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Actress ratio', fontsize=12)
    plt.xticks(gender_year_sum_df['Movie_release_date_x'][::5], rotation=45)  # Show every 5th year
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return standard_dev

def role_man(first_role):
    """
    Function to check if the first role is assigned to a man.
    
    Parameters:
    - first_role: string "M" or "F".
    
    Returns:
    - 1 if first role is for male, 0 otherwise.
    """
    if first_role == "M":
        return 1
    else:
        return 0


def calculate_female_percentage(genders):
    valid_genders = [g for g in genders if pd.notna(g)]
    if len(valid_genders) == 0:
        return -1
    female_count = sum(1 for g in valid_genders if g == 'F')
    return (female_count / len(valid_genders)) * 100


def preprocess_text(text):
    
    STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS.union({'film', 'story', 'character', 'characters', 'movie', 'movies'})

    if pd.isnull(text):
        return ""
    
    # Lowercase and remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Remove person names (PERSON entities)
    tokens = [token.lemma_ for token in doc 
              if token.ent_type_ != 'PERSON' 
              and token.lemma_ not in STOPWORDS 
              and token.lemma_.isalpha()]
    
    return ' '.join(tokens)

def get_top_n_similar_words(centroid, token_embeddings, top_n=50):

    # Stack all token embeddings into a matrix
    tokens = list(token_embeddings.keys())
    embeddings = np.vstack(list(token_embeddings.values()))
    
    # Compute cosine similarity between centroid and all token embeddings
    centroid_norm = centroid / np.linalg.norm(centroid) if np.linalg.norm(centroid) != 0 else centroid
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    similarity_scores = np.dot(embeddings_norm, centroid_norm)
    
    # Get top N indices
    top_n_idx = similarity_scores.argsort()[-top_n:][::-1]
    
    # Retrieve top N words and their similarity scores
    top_words = [(tokens[i], similarity_scores[i]) for i in top_n_idx]
    
    return top_words
