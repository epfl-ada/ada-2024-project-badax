
import json
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import umap
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
import warnings
from scipy.stats import mannwhitneyu
import seaborn as sns

# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

try:
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
except OSError:
    print("Model 'en_core_web_sm' not found. Downloading...")
    spacy.cli.download("en_core_web_sm", disable=['parser', 'ner'])
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    
def preprocess_movie_data(df_movies):
    # Convert release date to a datetime if not already
    df_movies['Movie_release_date'] = pd.to_datetime(df_movies['Movie_release_date'], errors='coerce')

    # Rename misnamed columns
    df_movies.rename(columns={'Movie_countxries': 'Movie_countries'}, inplace=True)

    # Country mapping dictionary
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
        'Iraqi Kurdistan': 'Iraq',
        'German Language': 'Germany',  # Language, assuming tied to Germany
        'Palestinian Territories': 'Palestine',
        'Kingdom of Italy': 'Italy',
        'Ukrainian SSR': 'Ukraine',
        'Republic of China': 'China',
        'Makau': 'China',
        'Aruba': 'Netherlands'
    }

    # Convert country names according to the mapping
    df_movies["Movie_countries"] = df_movies["Movie_countries"].apply(lambda x: list(json.loads(x).values()))
    df_movies["Movie_countries"] = df_movies["Movie_countries"].apply(
        lambda countries: [mapping.get(country, country) for country in countries]
    )

    # Drop rows without Clean_Summary or where Clean_Summary is too short
    df_movies = df_movies.dropna(subset=['Clean_Summary'])
    df_movies = df_movies[df_movies['Clean_Summary'].str.strip() != ""]

    # Tokenize each summary into a list of words
    df_movies["tokens"] = df_movies["Clean_Summary"].apply(lambda x: x.split())

    # Compute word and document frequencies
    word_in_docs = {}
    for tokens in df_movies["tokens"]:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            word_in_docs[token] = word_in_docs.get(token, 0) + 1

    # Set thresholds for filtering words
    too_common_threshold = 0.5 * len(df_movies)  # word appears in more than 50% of movies
    too_rare_threshold = 5  # word appears in less than 5 documents

    too_common_words = {w for w, count in word_in_docs.items() if count > too_common_threshold}
    too_rare_words = {w for w, count in word_in_docs.items() if count < too_rare_threshold}

    # Filter out too common or too rare words
    df_movies["tokens_filtered"] = df_movies["tokens"].apply(
        lambda tokens: [w for w in tokens if w not in too_common_words and w not in too_rare_words]
    )

    # Select the relevant columns to return
    return df_movies[['Wikipedia_movie_ID', 'Movie_name', 'Movie_countries', 'Movie_release_date', 'Female Percentage', 'tokens_filtered']]

def preprocess_and_save_data(df, preprocess_function, output_path):

    df['Clean_Summary'] = df['Movie_Summary'].apply(preprocess_function)

    print(f"Saving preprocessed data to {output_path}...")
    df.to_parquet(output_path, index=False)
    print("Preprocessed data saved successfully.")
    return df

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


def load_and_merge_data(metadata_loader, summaries_loader, characters_filepath):
    """
    Loads and merges movie metadata, summaries, and character data into a single DataFrame.

    Parameters:
        metadata_loader (callable): Function to load movie metadata.
        summaries_loader (callable): Function to load movie summaries.
        characters_filepath (str): File path to the character metadata TSV file.

    Returns:
        DataFrame: Merged DataFrame with movie and character information.
    """
    # Load CMU Corpus Dataset
    metadata = metadata_loader()
    summaries_df = summaries_loader()
    cmu_df = metadata.merge(summaries_df, on="Wikipedia_movie_ID")

    # Load characters data
    characters = pd.read_table(characters_filepath, header=None)
    characters.columns = [
        "Wikipedia_movie_ID",
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

    # Merging character database with summaries
    characters = characters[["Wikipedia_movie_ID", "Actor gender", "Character name"]]
    characters = characters.groupby("Wikipedia_movie_ID").agg({
        "Actor gender": list, 
        "Character name": list,
    }).reset_index()

    char_sum_CMU = pd.merge(cmu_df, characters, on="Wikipedia_movie_ID", how="inner")
    # Keeping only relevant columns
    char_sum_CMU = char_sum_CMU[["Wikipedia_movie_ID", "Movie_name", "Actor gender", "Movie_countxries","Movie_release_date" ,"Character name","Movie_Summary"]]

    # Create the new column with the female percentage
    char_sum_CMU['Female Percentage'] = char_sum_CMU['Actor gender'].apply(calculate_female_percentage)
    char_sum_CMU = char_sum_CMU[char_sum_CMU['Female Percentage'] != -1]

    print(f"Total movies combined: {char_sum_CMU.shape[0]}")
    return char_sum_CMU


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

def visualize_centroids_and_words(category_centroids, sentiment_lexicons, token_embeddings):
    # Collect centroids and their top-N similar words
    data = []
    for category, centroid in category_centroids.items():
        top_words = sentiment_lexicons[category]  # Set of top-N words
        for word in top_words:
            embedding = token_embeddings.get(word)
            if embedding is not None:
                data.append({
                    'category': category,
                    'word': word,
                    'embedding': embedding,
                    'is_centroid': False
                })
        # Also add the centroid itself
        data.append({
            'category': category,
            'word': 'Centroid',
            'embedding': centroid,
            'is_centroid': True
        })

    df_embeddings = pd.DataFrame(data)

    # Suppress the specific UMAP warning
    warnings.filterwarnings(
        "ignore",
        message="n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism."
    )

    # Dimensionality Reduction with UMAP
    reducer_3d = umap.UMAP(n_components=3, random_state=42)
    embeddings = np.stack(df_embeddings['embedding'].values)
    embedding_3d = reducer_3d.fit_transform(embeddings)

    # Assign 'x', 'y', 'z' Columns to df_embeddings
    df_embeddings[['x', 'y', 'z']] = embedding_3d

    # Create Copies of Sliced DataFrames
    df_centroids = df_embeddings[df_embeddings['is_centroid']].copy()
    df_words = df_embeddings[~df_embeddings['is_centroid']].copy()

    # Create 'display_category'
    df_words['display_category'] = df_words['category'].str.replace('_score', '', regex=False)
    df_centroids['display_category'] = df_centroids['category'].str.replace('_score', '', regex=False)

    # Define Categories and Color Mapping
    categories = pd.concat([
        df_words['display_category'],
        df_centroids['display_category']
    ]).unique()
    color_palette = px.colors.qualitative.Plotly
    color_discrete_map = {cat: color_palette[i % len(color_palette)] for i, cat in enumerate(categories)}

    # Clip Extreme Outliers to Improve Visibility
    x_min, x_max = np.percentile(df_words['x'], [1, 99])
    y_min, y_max = np.percentile(df_words['y'], [1, 99])
    z_min, z_max = np.percentile(df_words['z'], [1, 99])

    # Create 3D Plot
    fig_3d = go.Figure()
    for cat in categories:
        cat_df = df_words[df_words['display_category'] == cat]
        if not cat_df.empty:
            fig_3d.add_trace(
                go.Scatter3d(
                    x=cat_df['x'],
                    y=cat_df['y'],
                    z=cat_df['z'],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=color_discrete_map[cat],
                        opacity=0.7
                    ),
                    customdata=np.stack([cat_df['word'], cat_df['display_category']], axis=-1),
                    hovertemplate="Word: %{customdata[0]}<br>Category: %{customdata[1]}",
                    name=cat
                )
            )

    # Add centroid points
    for cat in categories:
        cat_centroids = df_centroids[df_centroids['display_category'] == cat]
        if not cat_centroids.empty:
            fig_3d.add_trace(
                go.Scatter3d(
                    x=cat_centroids['x'],
                    y=cat_centroids['y'],
                    z=cat_centroids['z'],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=8,
                        color=color_discrete_map[cat],
                        line=dict(width=2, color='black')
                    ),
                    customdata=np.stack([cat_centroids['word'], cat_centroids['display_category']], axis=-1),
                    hovertemplate="Word: %{customdata[0]}<br>Category: %{customdata[1]}",
                    showlegend=False
                )
            )

    # Layout Settings
    fig_3d.update_layout(
        title='Top-N Similar Words to Category Centroids (3D)',
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3',
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
            zaxis=dict(range=[z_min, z_max])
        ),
        legend_title='Category',
        hovermode='closest',
        width=700,
        height=500,
        margin=dict(l=0, r=0, b=0, t=50),
        scene_camera=dict(
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
    )

    # Display the 3D Plot
    fig_3d.show("svg")

    # Save the 3D Figure as an HTML File
    fig_3d.write_html("centroids_top_words_plot_3d.html")

def compute_sentiment_scores(tokens, sentiment_lexicons):
    scores = {}
    token_counts = defaultdict(int)
    for token in tokens:
        token_counts[token] += 1
    total_words = len(tokens)
    for cat, lexicon in sentiment_lexicons.items():
        count = sum(token_counts[word] for word in lexicon if word in token_counts)
        scores[cat] = count / total_words if total_words > 0 else 0
    return scores

def calculate_sentiment_for_dataframe(df, sentiment_lexicons):

    # Pre-compute sentiment scores for all rows
    all_scores = [
        compute_sentiment_scores(tokens, sentiment_lexicons)
        for tokens in df['tokens_filtered']
    ]

    # Convert the list of dictionaries to a DataFrame
    sentiment_scores_df = pd.DataFrame(all_scores)

    # Combine sentiment scores with the original DataFrame
    df_combined = pd.concat([df.reset_index(drop=True), sentiment_scores_df.reset_index(drop=True)], axis=1)
    return df_combined

def plot_sentiment_comparison(sentiment_stats, sentiment_cols, title='Comparison of Sentiment Scores by Gender Presence in Movies'):
    # Create a list of formatted category names (remove "_score_proximity", capitalize)
    formatted_categories = [col.replace("_score", "").capitalize() for col in sentiment_cols]

    # Create the figure
    fig = go.Figure()

    # Add trace for female movies
    fig.add_trace(go.Bar(
        x=formatted_categories,
        y=sentiment_stats['Female Movies Mean'],
        error_y=dict(
            type='data',
            array=sentiment_stats['Female Movies SEM'],
            visible=True
        ),
        name='≥ 66% Female',
        marker_color='salmon',
        hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<extra>≥ 66% Female</extra>',
    ))

    # Add trace for male movies
    fig.add_trace(go.Bar(
        x=formatted_categories,
        y=sentiment_stats['Male Movies Mean'],
        error_y=dict(
            type='data',
            array=sentiment_stats['Male Movies SEM'],
            visible=True
        ),
        name='≤ 33% Female',
        marker_color='skyblue',
        hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<extra>≤ 33% Female</extra>',
    ))

    # Update layout for better readability and grouping
    fig.update_layout(
        title=title,
        xaxis_title='Sentiment Category',
        yaxis_title='Average Sentiment Score',
        barmode='group',
        bargroupgap=0.1,
        hovermode='x',
        template='plotly_white',
        width=700,
        height=500
    )

    # Enable gridlines on the y-axis
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Show the figure
    fig.show("svg")

    # Optionally, save as an HTML file for embedding on a website
    # fig.write_html("interactive_sentiment_comparison.html")

def perform_stat_tests(female_movies, male_movies, sentiment_cols, female_means, male_means, alpha=0.05):
    # Function to perform statistical tests and return the result
    def perform_stat_tests(female_data, male_data):
        # Perform Mann-Whitney U test (non-parametric)
        stat, p = mannwhitneyu(female_data, male_data, alternative='two-sided')
        return stat, p

    # Initialize an empty list to collect test results
    test_results = []

    # Iterate over each sentiment to perform tests and check significance
    for sentiment in sentiment_cols:
        stat, p = perform_stat_tests(female_movies[sentiment], male_movies[sentiment])
        is_significant = "Yes" if p < alpha else "No"
        # Determine direction of significance, if significant
        direction = 'female' if female_means[sentiment] > male_means[sentiment] and is_significant == "Yes" else 'male'
        # Collect each row of results
        test_results.append({
            'Sentiment': sentiment,
            'U-Statistic': stat,
            'P-Value': p,
            'Significant': is_significant,
            'Direction': direction if is_significant == "Yes" else 'N/A'  # Only add direction if significant
        })

    # Create a DataFrame from the results
    df_results = pd.DataFrame(test_results)

    return df_results

def generate_sentiment_comparison_markdown(female_means, male_means, female_movies, male_movies, sentiment_cols, output_filename="sentiment_comparison.md", alpha=0.05):
 
    rows = []
    rows.append("| Category | Female Mean | Male Mean | p-value | Significant? | Higher in |")
    rows.append("|----------|-------------|-----------|---------|--------------|-----------|")

    for sentiment in sentiment_cols:
        female_mean = female_means[sentiment]
        male_mean = male_means[sentiment]
        stat, p = mannwhitneyu(female_movies[sentiment], male_movies[sentiment], alternative='two-sided')
        
        # Determine significance
        significant = "Yes" if p < alpha else "No"
        
        # Determine direction if significant
        direction = ""
        if p < alpha:
            if female_mean > male_mean:
                direction = "Female Majority"
            else:
                direction = "Male Majority"
        else:
            direction = "-"
        
        category_name = sentiment.replace("_score", "").capitalize()
        row = f"| {category_name} | {female_mean:.3f} | {male_mean:.3f} | {p:.5f} | {significant} | {direction} |"
        rows.append(row)

    # Join all rows into a markdown content
    markdown_content = "\n".join(rows)

    # Save to a markdown file
    with open(output_filename, "w") as f:
        f.write(markdown_content)

    print(f"Markdown table saved to {output_filename}")

def plot_sentiment_trends_over_time(df, top_sentiments, movie_release_date_col='Movie_release_date', female_percentage_col='Female Percentage'):
    """
    Plots trends of top sentiments over time categorized by high and low female percentage in movies.

    Parameters:
        df (DataFrame): DataFrame containing the movie data.
        top_sentiments (list): List of column names for the sentiments to plot.
        movie_release_date_col (str): Column name for the movie release dates.
        female_percentage_col (str): Column name for the female percentage in movies.
    """
    # Add a 'Year' column if the release date is in a datetime format
    df['Year'] = pd.to_datetime(df[movie_release_date_col]).dt.year

    # Add a 'Subset' column to differentiate male-majority and female-majority movies
    df['Subset'] = df.apply(
        lambda row: 'High Female % (≥66%)' if row[female_percentage_col] >= 66 else (
            'Low Female % (≤33%)' if row[female_percentage_col] <= 33 else None
        ),
        axis=1
    )

    # Filter out rows not part of the male or female subsets
    df_filtered = df[df['Subset'].notnull()]

    # Define colors for the subsets
    subset_colors = {
        'High Female % (≥66%)': 'salmon',
        'Low Female % (≤33%)': 'skyblue'
    }

    # Create subplots for all sentiments
    num_sentiments = len(top_sentiments)
    fig, axes = plt.subplots(
        nrows=(num_sentiments + 2) // 3,
        ncols=3,
        figsize=(18, 12),
        constrained_layout=True
    )
    axes = axes.flatten()

    for idx, sentiment in enumerate(top_sentiments):
        if idx >= len(axes):
            break
        
        # Plot on the specified subplot axis
        for subset, color in subset_colors.items():
            subset_data = df_filtered[df_filtered['Subset'] == subset]
            sns.regplot(
                x='Year',
                y=sentiment,
                data=subset_data,
                scatter=False,
                ax=axes[idx],
                color=color,
                label=subset,
                line_kws={'linewidth': 2}
            )
        
        axes[idx].set_title(f'Trend of {sentiment.replace("_proximity", "").capitalize()} Over Time', fontsize=12)
        axes[idx].set_xlabel('Year', fontsize=10)
        axes[idx].set_ylabel('Average Sentiment Score', fontsize=10)
        axes[idx].grid(visible=True)

    # Remove unused subplots
    for idx in range(len(top_sentiments), len(axes)):
        fig.delaxes(axes[idx])

    # Add a legend outside the subplots
    fig.legend(
        labels=subset_colors.keys(),
        loc='lower right',
        ncol=2,
        fontsize=12
    )

    # Set the main title for the figure
    fig.suptitle('Semantic Category Trends by Female Percentage', fontsize=16)

    plt.show()

def create_image_selector_html(image_dir):
    """
    Generates an HTML file that allows users to select and view images from a specified directory.

    Parameters:
        image_dir (str): Directory containing image files to display in the HTML selector.
    """
    # Get all PNG files in the specified directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    image_files.sort()  # Sort for consistent ordering

    # We'll pick the first image as the default if it exists
    default_image = image_files[0] if image_files else ''

    # Construct the HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Image Selector</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; }}
  #image-container {{
    margin-top: 20px;
  }}
  img {{
    max-width: 100%;
    height: auto;
  }}
</style>
</head>
<body>

<h1>Select an Image:</h1>
<select id="imageSelect">
"""

    # Add an option for each image
    for i, img in enumerate(image_files):
        selected_attr = 'selected' if i == 0 else ''
        html_content += f'  <option value="{image_dir}/{img}" {selected_attr}>{img}</option>\n'

    html_content += f"""</select>

<div id="image-container">
  <img id="displayImage" src="{image_dir}/{default_image}" alt="Selected Image"/>
</div>

<script>
  const imageSelect = document.getElementById('imageSelect');
  const displayImage = document.getElementById('displayImage');

  imageSelect.addEventListener('change', function() {{
    const selectedSrc = imageSelect.value;
    displayImage.src = selectedSrc;
  }});
</script>

</body>
</html>
"""

    # Save the HTML to a file
    html_file_path = os.path.join(image_dir, "image_selector.html")
    with open(html_file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML file created: {html_file_path}")

def create_sentiment_choropleth(country_comparison, sentiment_cols):
    """
    Generates an interactive choropleth map displaying sentiment scores across countries
    for female-majority and male-majority movies.

    Parameters:
        country_comparison (DataFrame): DataFrame containing the country and sentiment data.
        sentiment_cols (list): List of sentiment column names to visualize.
    """
    # Initialize a list to hold all the choropleth traces
    data_traces = []

    for i, sentiment in enumerate(sentiment_cols):
        # Prepare data for female-majority movies
        data_to_plot_female = country_comparison[['Movie_countries', f'{sentiment}_female']].dropna()
        data_to_plot_female.rename(columns={
            'Movie_countries': 'country',
            f'{sentiment}_female': 'Female_Majority_Score'
        }, inplace=True)

        # Prepare data for male-majority movies
        data_to_plot_male = country_comparison[['Movie_countries', f'{sentiment}_male']].dropna()
        data_to_plot_male.rename(columns={
            'Movie_countries': 'country',
            f'{sentiment}_male': 'Male_Majority_Score'
        }, inplace=True)

        # Merge female and male data
        data_to_plot = data_to_plot_male.merge(data_to_plot_female, on='country', how='inner')
        data_to_plot['Female_majority'] = data_to_plot['Female_Majority_Score'] > data_to_plot['Male_Majority_Score']
        
        # Define a custom color scale based on the Female_majority column
        data_to_plot['color'] = data_to_plot['Female_majority'].apply(lambda x: 'salmon' if x else 'skyblue')

        # Create custom hovertext
        data_to_plot['hover_text'] = (
            "Country: " + data_to_plot['country'] +
            "<br>Male Majority Score: " + data_to_plot['Male_Majority_Score'].astype(str) +
            "<br>Female Majority Score: " + data_to_plot['Female_Majority_Score'].astype(str) +
            "<br>Sentiment: " + sentiment
        )

        # Update the trace
        choropleth_trace = go.Choropleth(
            locations=data_to_plot['country'],
            locationmode='country names',
            z=data_to_plot['Female_majority'].astype(int),
            colorscale=[[0, 'skyblue'], [1, 'salmon']],
            showscale=False,  # Disable the color bar
            text=data_to_plot['hover_text'],  # Use custom hover text
            hoverinfo='text'  # Display only the custom hover text
        )
        data_traces.append(choropleth_trace)

    # Create dropdown menu buttons
    buttons = []
    for i, sentiment in enumerate(sentiment_cols):
        button = dict(
            label=sentiment.replace("_proximity", "").capitalize(),
            method='update',
            args=[
                {'visible': [j == i for j in range(len(sentiment_cols))]},  # Toggle visibility
                {'title': f'Majority Movies: {sentiment.replace("_proximity", "").capitalize()} Scores'}
            ]
        )
        buttons.append(button)

    # Layout with improved legend
    layout = go.Layout(
        title=f'Majority Movies: {sentiment_cols[0].replace("_proximity", "").capitalize()} Scores',
        updatemenus=[dict(
            buttons=buttons,
            direction='down',
            showactive=True,
            x=0.5,
            y=1.15,
            xanchor='left',
            yanchor='top'
        )],
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='black',
            showcountries=True,
            countrycolor='black',
            projection_type='equirectangular',
        ),
        margin=dict(l=10, r=10, t=100, b=10),
        height=600,
        width=700,
        annotations=[
            dict(
                x=0.05,  # Position for "Female Majority"
                y=1.02,
                xref="paper",
                yref="paper",
                text='<span style="color:salmon;">&#9632;</span> Female Majority',
                showarrow=False,
                font=dict(size=12),
            ),
            dict(
                x=0.05,  # Position for "Not Female Majority"
                y=0.98,
                xref="paper",
                yref="paper",
                text='<span style="color:skyblue;">&#9632;</span> Male Majority',
                showarrow=False,
                font=dict(size=12),
            ),
        ],
    )

    # Create the figure with all traces and the defined layout
    fig = go.Figure(data=data_traces, layout=layout)

    # Display the interactive figure
    fig.show("svg")
    # fig.write_html("geographic_sentiment_score.html")
