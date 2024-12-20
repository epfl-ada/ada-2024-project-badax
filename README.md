# Screening Change: Feminism, Representation, and Social Transformation in Film

## Abstract

We believe that the movie industry reflects society’s main values and concerns. Movies can provide new ideals, provide realistic historical depictions (i.e war/slavery), or address avant-gardist themes to shape and improve the future of our world. Feminism is a topic that lies in between blockbusters (because it concerns the globality of the world) but remains well under-represented and remains a hot topic. Understanding the transformation of society by feminism through films is a way to grasp the dynamics of other avant-garde/progressist topics in society. 
Primarily, this project studies the presence and the role of women in the movie industry. The second part of the project focuses on the evolution of women’s roles in erotic, war, and religious movies. Lastly, this project studies whether directors who tend to be more inclusive towards female actors, also tackle other societal issues like climate change.

## Research Questions:
### First part: Presence and role of women in the movie industry
- What is the actor-gender ratio in movies throughout the years? What is the evolution of that ratio for the leading roles? 
- Are the ratios globally different across different countries/regions? Are some countries/regions more biased against women?
- How are women portrayed in the film (strong, smart, powerful) or presented as secondary characters (solely physical trait characterization, non-charismatic, based on stereotypes)? 
- How are the reviews of movies that empower women? Can we identify patterns for the viewers? Are they usually supportive, unsupportive, or neither?
### Second part: Women in erotic, war, and religious movies, three genres highly non-inclusive 
- How is the evolution of women's presence in these genres? 
- How do their roles evolve over time (backline vs dominant characters)? 
### Third part : Inclusive directors on progressist topics
- What is the parity ratio between men and women among directors? Do films with women directors describe female characters less stereotypically?
- Do films with better women representation also tend to tackle more important issues like climate change? 
- Can we identify patterns (gender ratios, character description, movie plots, etc.) with films that are more inclusive towards women? 

## Additional Datasets

- **[IMDB](https://datasets.imdbws.com/)**: Provides detailed information about movie crew, cast, directors, and ratings.  
- **[Rotten Tomato Reviews](https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews)**: Contains reviews of movies, enabling the analysis of viewers' reactions.  
- **[Wikidata](https://query.wikidata.org/)**: Facilitates merging datasets by matching IMDB and Freebase Movie IDs.  

We can split methods to these subroutines:

## 1. Data Integration
#### Merge CMU corpus and IMDB datasets using IMDB ID to Freebase ID via Wikidata:
- This ensures the linking of metadata across datasets. The use of Freebase ID acts as a robust intermediary key, however with a tradeoff in the form of losing some movies
IMDB's detailed information about cast and gender, along with a clear hierarchy of primary actors, will enhance the CMU corpus data and allow us to get better insights from the data.
- In order to prevent losing information, we could also get data from TMDB, try getting more information from Wikidata, or try data scraping.
## 2. Exploratory Data Analysis
#### Visualize trends of parity over time:
- This helps identify historical patterns and changes in gender representation.
- Visualization (line graphs, bar plots, etc.) will show trends like the female ratio in movies over decades, gender distribution in lead roles, or genres dominated by one gender.
#### Visualize trends based on countries/regions:
- It allows us to understand how women are represented based on the region. 
## 3. Text Analysis
#### NLP:
- Applying NLP-based text analysis (like BERT model) to CMU plot summaries and potentially Rotten Tomatoes to discover movies’ biases to later connect them with the situation of women
- A fine-tuned BERT model could extract sentiment, recurring themes, or portrayals of women characters, enabling in-depth quantitative analysis after assigning a numerical score to a sentiment or other labels (like the violence toward women present in the movie).
#### Explicit keyword search in summaries and reviews:
- This could potentially be used to support the NLP-based solution, the idea is to look for some specific words in plot summaries to classify them (for example, we could classify that a movie is violent if it has some words from a set of words associated with violence). Not as efficient as the previous method, can be potentially used as a supporting method.
## 4. Hypothesis Testing
- Statistical analysis to test the relation between women’s leading roles and movie release date. Test whether there’s a significant increase in female lead roles over time.
- Compare sentiment analysis scores for movies with a male lead versus a female lead to determine any significant differences in perception.

## Proposed timelines:
- Finish Initial gender representation analysis (part of research questions): Nov 22
- If feasible, analyze all research questions with any additional datasets : graphs  Nov 29
- Start working on a website (creation, putting part of analysis in form of a story) : Dec 6
- Finishing the website: Dec 13th
- P3 Due (polishing all jupyter notebooks, github repo, and website): Dec 20


 ## Organisation: 
- Oskar: Cleaning and merging datasets, map creation for revenue analysis
- Yassine: Exploratory data analysis with statistical testing, analysis of female directors
- Alessandro: Analysis on war/violent movies and relation with feminism
- Lorentz: Overall gender analysis, writing up report on data story
- Alix: Application of NLP to gender description analysis, writing up report on data story

## How to run the notebook: 
Make sure to download the datasets above into a “data” folder in order to run the notebook 

