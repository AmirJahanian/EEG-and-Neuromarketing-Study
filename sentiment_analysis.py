# The code below uses NLTK's VADER sentiment analysis tool to predict 
#customer decision of liking/disliking a product based on comments about them
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pandas as pd


# Importing data from an excel file:
data = pd.read_excel('rs.xlsx', sheet_name='Sheet1') # modify the file name to get your input
# Defining the analyzer:
nltk.download('vader_lexicon')
analyzing = SentimentIntensityAnalyzer()

# Calculating sentiment scores and classifying them
scores = []
for text in data['feedback']:
    scores.append(analyzing.polarity_scores(text)['compound'])

#Defining the boundaries
sentiment_class = ['like' if score > 0.5 else 'dislike' if score < -0.5 else 'neutral' for score in scores]

# Writing the scores
data['sentiment_score'] = scores
data['predicted_decision'] = sentiment_class

#Exporting results as an excel sheet
data.to_excel('results.xlsx', index=False) # replace with a desired output file name
