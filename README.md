
# Report: Analysis of Twitter Sentiment Data

# Introduction:
# This report summarizes the steps taken to analyze Twitter sentiment data and the observations derived from the process.
# The objective was to load, preprocess, and perform exploratory data analysis on the provided datasets.

# Data Loading and Merging:
# 1. Two datasets, 'twitter_training.csv' and 'twitter_validation.csv', were uploaded using `google.colab.files.upload()`.
# 2. These datasets were loaded into pandas DataFrames.
# 3. The columns were renamed to 'Id', 'Entity', 'Sentiment', and 'Text' for clarity.
# 4. The two DataFrames were concatenated into a single DataFrame named `data` for unified processing.

# Initial Data Inspection and Cleaning:
# 1. The dimensions (`shape`) and column information (`info()`) of the combined DataFrame were examined.
# 2. Descriptive statistics (`describe()`) were generated for the 'Entity', 'Sentiment', and 'Text' columns to understand their distributions.
#    - Observation: The value counts for 'Entity' and 'Sentiment' revealed the distribution of different entities and sentiment labels in the combined dataset.
# 3. Missing values were identified using `isnull().sum()`.
#    - Observation: There were missing values primarily in the 'Text' column.
# 4. The entities associated with missing 'Text' data were identified.
# 5. The count of missing 'Text' values per 'Entity' and 'Sentiment' combination was analyzed.
# 6. A function `fill_missing_text` was implemented to address missing 'Text' values. This function attempts to fill missing text for a specific entity and sentiment combination using the most frequent non-null text within that group.
# 7. The missing text values were filled using this function.
#    - Observation: The `isnull().sum()` check after filling confirmed that missing 'Text' values were handled.

# Text Preprocessing:
# 1. Necessary NLTK resources ('punkt', 'stopwords', 'vader_lexicon') were downloaded.
# 2. A `remove_emojis` function was defined using regular expressions.
# 3. A comprehensive `preprocess_text` function was created to perform:
#    - Lowercasing the text.
#    - Removing URLs.
#    - Removing emojis.
#    - Removing punctuation and special characters (keeping only letters and spaces).
#    - Tokenizing the text into words.
#    - Removing English stopwords.
#    - Joining the filtered tokens back into a cleaned string.
# 4. The `preprocess_text` function was applied to the 'Text' column to create a new 'cleaned_review' column.
#    - Observation: The head of the DataFrame showed the comparison between the original 'Text' and the 'cleaned_review', demonstrating the effects of preprocessing.
# 5. The 'Id' and 'Entity' columns were dropped as they were no longer needed for sentiment analysis.
# 6. An alternative, simpler text cleaning step was also performed on the 'Text' column and assigned back to 'cleaned_review', focusing on lowercasing and removing non-alphanumeric characters.
#    - Observation: This step seems to overwrite the more detailed preprocessing performed earlier. The report assumes the later step was the intended final cleaning for the 'cleaned_review' column.

# Sentiment Analysis with VADER:
# 1. The VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer was initialized.
# 2. A function `vader_sentiment_classify` was defined to categorize sentiment into 'positive', 'negative', or 'neutral' based on the VADER compound score.
# 3. The VADER compound score was calculated for each 'cleaned_review' and stored in a new column 'compound_score'.
# 4. The `vader_sentiment_classify` function was applied to the 'cleaned_review' to assign a 'vader_sentiment' label to each entry.
#    - Observation: The head of the DataFrame showed the calculated compound scores and the assigned VADER sentiment labels.

# Tokenization and Word Frequency Analysis:
# 1. A `tokenize_text` function was defined to convert cleaned text into a list of tokens (words) using `nltk.word_tokenize`.
# 2. This function was applied to the 'cleaned_review' column to create a 'tokens' column containing lists of tokens.
#    - Observation: The head of the DataFrame showed the tokenized lists for each cleaned review.
# 3. The frequency of tokens across the entire dataset was counted using `collections.Counter`.
# 4. The top 20 most frequent tokens were identified.
# 5. A bar chart was created using Plotly to visualize the top 20 most frequent tokens and their frequencies.
#    - Observation: The chart displayed the most common words appearing in the dataset after cleaning and stopword removal.

# Sentiment Distribution Visualization:
# 1. The counts of each sentiment category from the 'vader_sentiment' column were obtained.
# 2. A bar plot was created using Plotly to visualize the distribution of 'positive', 'negative', and 'neutral' sentiments as classified by VADER.
#    - Observation: This plot provides a clear overview of the sentiment balance in the dataset according to VADER.

# Word Cloud Visualization by Sentiment:
# 1. A function `get_text_by_sentiment` was defined to concatenate all cleaned reviews for a given sentiment category.
# 2. A function `plot_wordcloud` was defined to generate and display a word cloud from a given text string.
# 3. Word clouds were generated and plotted separately for 'negative', 'neutral', and 'positive' sentiments based on the VADER sentiment labels.
#    - Observation: These word clouds visually highlight the most prominent words associated with each sentiment category, offering insights into the language used in positive, negative, and neutral reviews.

# Analysis of Unique Tokens by Sentiment:
# 1. A function `get_unique_token_counts` was created to find tokens that are common within a target sentiment category but less common or absent in other sentiment categories.
# 2. The top 20 unique tokens for 'negative' sentiment (vs 'neutral') and 'neutral' sentiment (vs 'negative') were identified.
# 3. A subplot with two bar charts was created using Plotly to visualize the top unique tokens for 'negative' and 'neutral' sentiments.
#    - Observation: This visualization helps identify words that are particularly characteristic of negative or neutral reviews, distinguishing them from the other sentiment.
# 4. The top 20 unique tokens for 'positive' sentiment (vs 'neutral' and 'negative') were identified.
# 5. A bar chart was created using Plotly to visualize these unique positive tokens.
#    - Observation: This chart highlights words that are most distinctively associated with positive reviews.

# Summary of Observations:
# - The datasets were successfully loaded, merged, and inspected.
# - Missing 'Text' data was identified and handled by imputing with the most frequent text for specific entity/sentiment combinations.
# - Text preprocessing involved lowercasing, removing URLs, emojis, punctuation, and stopwords, resulting in a 'cleaned_review' column.
# - VADER sentiment analysis was performed, assigning sentiment labels ('positive', 'negative', 'neutral') and compound scores.
# - The distribution of VADER-classified sentiments was visualized, showing the count of each category.
# - Word clouds for each sentiment provided a visual summary of frequent terms within positive, negative, and neutral reviews.
# - Analysis of unique tokens helped pinpoint words that are more characteristic of specific sentiment categories, aiding in understanding the linguistic nuances of each sentiment.

# Next Steps (Implied from imports but not executed in the provided code):
# - Machine Learning Model Training: The presence of imports like `train_test_split`, `LabelEncoder`, `MinMaxScaler`, `LogisticRegression`, etc., suggests an intention to build a supervised machine learning model for sentiment classification based on the preprocessed text data and potentially the original sentiment labels.
# - Evaluation: Imports for evaluation metrics like ROC AUC and confusion matrix indicate plans for assessing the performance of a trained model.
