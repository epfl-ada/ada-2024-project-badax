import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json 
from collections import Counter
import re
import ast
import seaborn as sns
import spacy
from collections import Counter
import re
from wordcloud import WordCloud
import multiprocessing as mp

# Function to calculate female percentage
def calculate_female_percentage(genders):
    valid_genders = [g for g in genders if pd.notna(g)]
    if len(valid_genders) == 0:
        return -1
    female_count = sum(1 for g in valid_genders if g == 'F')
    return (female_count / len(valid_genders)) * 100

def plot_female_percentage_graph(char_sum_CMU):
    # Create the histogram with a cool-to-warm gradient
    plt.figure(figsize=(10, 6))

    # Define a cool-to-warm colormap
    cmap = plt.cm.coolwarm

    # Normalize the data for color mapping
    norm = plt.Normalize(char_sum_CMU["Female Percentage"].min(), char_sum_CMU["Female Percentage"].max())

    # Plot the histogram
    n, bins, patches = plt.hist(
        char_sum_CMU["Female Percentage"],
        bins=30,
        edgecolor='black'
    )

    # Apply the cool-to-warm gradient to the bars
    for bin, patch in zip(bins[:-1], patches):
        color = cmap(norm(bin))
        patch.set_facecolor(color)

    # Add KDE using Seaborn
    sns.kdeplot(char_sum_CMU["Female Percentage"], color='black', linewidth=2)

    # Add title, labels, and grid
    plt.xlabel('Percentage of Females in Cast', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Female Percentage in Casts', fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save and display the plot
    plt.savefig('data/figs/gender_prop_hist_gradient.png', dpi=300, bbox_inches='tight')
    plt.show()

# # Function to apply extract_adj_verbs in parallel using multiprocessing
def parallel_apply_multiprocessing(series, func, n_cores=4):
    with mp.Pool(n_cores) as pool:
        result = pool.map(func, series)
    return result

# Functions to extract adjectives and verbs from a text
def extract_adj(text):
    # Load the spaCy language model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Model 'en_core_web_sm' not found. Downloading...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.pos_ in ('ADJ')]

def extract_verb(text):
        # Load the spaCy language model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Model 'en_core_web_sm' not found. Downloading...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.pos_ in ('VERB')]
