# Topic Modeling and Recommendation of Reddit Communities

## Capstone project for Galvanize data science immersive program

## Project Goals:
The goal of this project is to give recommendations on reddit content (in the form of subreddits to follow) based on the kind of content a user already enjoys. In this project that will take the form of a web page where a user will input their Twitter handle and receive recommendations for subreddits to follow based on that data.

## Methods:
I began by getting data from the 100 top posts(or as many as there were) of around 600 popular subreddits. This data came in the form of titles, text from within text posts and comments. After pulling data from the reddit API I had around 6 million reddit commnents, titles and posts from which to draw text data from.

To featurize this data I used Keras to implement and train a word2vec pipeline and transform every word into a numerical vector with dimension 300.
