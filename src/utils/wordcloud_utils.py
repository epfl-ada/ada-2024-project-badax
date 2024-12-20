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
import pickle
import os
from matplotlib.colors import LinearSegmentedColormap
from itertools import chain
from scipy.stats import pearsonr, spearmanr


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

def create_counter(col):
    flat = np.array(list(chain.from_iterable(col)))
    return Counter(flat)

def filter_similar_counts(counter1, counter2, threshold=0.4):
    # Calculate total counts for each Counter to normalize
    total_count1 = sum(counter1.values())
    total_count2 = sum(counter2.values())

    # Calculate relative frequencies
    relative_freqs1 = {word: count / total_count1 for word, count in counter1.items()}
    relative_freqs2 = {word: count / total_count2 for word, count in counter2.items()}

    # Initialize filtered counters
    filtered_counter1 = Counter()
    filtered_counter2 = Counter()

    # Get the intersection of words from both counters
    common_keys = set(relative_freqs1.keys()).intersection(set(relative_freqs2.keys()))

    for word in common_keys:
        freq1 = relative_freqs1[word]
        freq2 = relative_freqs2[word]

        # Calculate the percentage difference between relative frequencies
        if abs(freq1 - freq2) / max(freq1, freq2) > threshold:
            # If the difference is greater than threshold, keep the word
            filtered_counter1[word] = counter1[word]
            filtered_counter2[word] = counter2[word]

    # Add words that are unique to each counter
    unique_keys1 = set(counter1.keys()).difference(set(counter2.keys()))
    unique_keys2 = set(counter2.keys()).difference(set(counter1.keys()))

    for word in unique_keys1:
        filtered_counter1[word] = counter1[word]

    for word in unique_keys2:
        filtered_counter2[word] = counter2[word]

    return filtered_counter1, filtered_counter2

# Create a mask for the word cloud (ellipse-shaped mask)
def create_ellipse_mask(width, height):
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width / 2, height / 2
    mask = (x - center_x) ** 2 / (width / 2) ** 2 + (y - center_y) ** 2 / (height / 2) ** 2 <= 1
    # Invert the mask (255 inside the ellipse, 0 outside)
    return 255 - 255 * mask.astype(np.uint8)

def gen_wordcloud(counts,gender):
    ellipse_mask = create_ellipse_mask(400, 400)
    if gender == 'male':
        colors = LinearSegmentedColormap.from_list(
            "adjusted_male", ["#1E90FF", "#87CEFA", "#B0C4DE", "#0000CD", "#000080"]
        )
        contour = 'blue'
    else:
        colors = LinearSegmentedColormap.from_list(
            "adjusted_female", ["#FF1493", "#FF69B4", "#F8BFD9", "#C71585", "#8B008B"]
        )
        contour = 'hotpink'
        

    # Generate the word cloud for female adjectives
    wordcloud_fem = WordCloud(
        width=400, height=400,
        background_color='white',
        colormap=colors,
        mask=ellipse_mask,
        contour_width=2,
        contour_color=contour
    ).generate_from_frequencies(counts)

    # Display the female word cloud
    plt.figure(figsize=(6, 6))
    plt.imshow(wordcloud_fem, interpolation='bilinear')
    plt.axis('off')
    # plt.title('Most Common Adjectives for Female Dominant Films', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/figs/female_adj_wordcloud.png')
    plt.show()

def plot_gender_vs_revenue(char_sum_CMU_original):
    # Plot the distribution
    char_sum_CMU = char_sum_CMU_original.copy()
    char_sum_CMU = char_sum_CMU.dropna(subset=['Movie_box_office_revenue'], how='any')
    # Bin the Female Percentage into groups using linspace
    bins = np.linspace(0, 100, 11)  # Create 10 bins from 0 to 100
    labels = [f'{int(bins[i])}-{int(bins[i+1])}%' for i in range(len(bins) - 1)]
    char_sum_CMU['Female_Percentage_Bins'] = pd.cut(char_sum_CMU['Female Percentage'], bins=bins, labels=labels, include_lowest=True)

    # Create a custom blue-to-pink gradient palette
    palette = sns.color_palette("coolwarm", len(labels))

    # Create the boxplot with Seaborn
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x='Female_Percentage_Bins',
        y='Movie_box_office_revenue',
        data=char_sum_CMU,
        showfliers=False,  # Hide outliers for cleaner visuals
        palette=palette,  # Aesthetic color palette
        linewidth=1.5,
        hue = 'Female_Percentage_Bins',
        legend=False
    )

    # Apply log scale to the revenue axis
    plt.yscale('log')

    # Customize the plot
    plt.title('Revenue vs Female Percentage', fontsize=18, fontweight='bold')
    plt.xlabel('Female Percentage (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Movie Box Office Revenue (Log Scale)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Add a legend or annotations if required
    plt.tight_layout()
    plt.savefig('revenue_vs_female_percentage.png', dpi=300, bbox_inches='tight')
    plt.show()

def hypothesis_test(char_sum_CMU):
    # Clean data: Remove rows with NaN values in the relevant columns
    cleaned_data = char_sum_CMU.copy()
    cleaned_data = cleaned_data.dropna(subset=['Female Percentage', 'Movie_box_office_revenue'])

    # Log-transform revenue for normalization (optional but helps with skewed data)
    cleaned_data['Log_Revenue'] = np.log(cleaned_data['Movie_box_office_revenue'] + 1)

    # Perform Pearson correlation test
    pearson_corr, pearson_p_value = pearsonr(cleaned_data['Female Percentage'], cleaned_data['Log_Revenue'])

    # Perform Spearman correlation test
    spearman_corr, spearman_p_value = spearmanr(cleaned_data['Female Percentage'], cleaned_data['Log_Revenue'])

    # Output results
    print(f"Pearson Correlation: {pearson_corr:.3f}, p-value: {pearson_p_value:.3e}")
    print(f"Spearman Correlation: {spearman_corr:.3f}, p-value: {spearman_p_value:.3e}")

    # Interpret results
    alpha = 0.05  # Significance level
    if pearson_p_value < alpha:
        print("The Pearson test indicates a significant relationship between gender ratio and revenue.")
    else:
        print("The Pearson test does not indicate a significant relationship.")

    if spearman_p_value < alpha:
        print("The Spearman test indicates a significant relationship between gender ratio and revenue.")
    else:
        print("The Spearman test does not indicate a significant relationship.")