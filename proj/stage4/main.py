import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# names = [userId#, movieId#, rating, timestamp]
df = pd.read_csv('ratings.csv')
df.head()

# names = [movieId#, title, genres]
movie_titles = pd.read_csv('movies.csv')
movie_titles.head()

# merge the ratings.csv and movies.csv on similiar "movieId"
df = pd.merge(df, movie_titles, on='movieId')
df.head()
df.describe()

# userId - The ID of the user who rated the movie.
# movieId - The ID of the movie.
# rating - The rating the user gave the movie, between 0.5 and 5.0
# timestamp - The time the movie was rated, starting midnight (UTC) of January 1, 1970.
# title - The title of the movie.
# genres - The type of movie

# (1) Get Average Rating and (2) Number of Ratings FOR EACH Movie
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())  # (1)
ratings.head()

# (3) Both Ratings will be used to calculate correlation between movies
# High Correlation = movies that are most similar, between -1 and 1.1
ratings['total_ratings'] = df.groupby('title')['rating'].count()
ratings.head()

# visulize the distribution of the Ratings (see Plots tab)
ratings['rating'].hist(bins=50) # most people chose 3, 3.5 or 4 (see peak)
                                    # and most ratings were between 3 and 4 (see spread)

# visualize the distribution of the Number of Ratings
ratings['total_ratings'].hist(bins=60) # popular movies have most ratings, (see peak)
                                       # but most movies have few ratings (see spread)

# Check Relationship between Average Rating for a Movie and the Number of Ratings it recieved.
import seaborn as sns
sns.jointplot(x='rating', y='total_ratings', data=ratings)

# The Plot shows a positive relationship between the average ratings
# of a movie and the number of rationgs. Also, it shows that the more
# ratings a movie gets the higher the average is.
# This would be important when choosing a threshold per the number
# of ratings per movie.

# Item based Recommendation System
# Task: use df.pivot_table to Convert dataset into Matrix using 
# Movie-Title and  user-id as the index, and ratings as the value.
# Result is a dataframe with the columns as a the movie title, rows as use id.
matrix = df.pivot_table(index='userId', columns='title', values='rating')
matrix.head()

# find the movies that have the most ratings by setting ascending to False, then view top 10
mostRated = ratings.sort_values('total_ratings', ascending=False).head(10)

# New Dataframes showing the userIds' and ratings that were given for both movies.
Forrest_Gump_rating = matrix['Forrest Gump (1994)']
Toy_Story_rating = matrix['Toy Story (1995)']

Forrest_Gump_rating.head()
Toy_Story_rating.head()

# Goal is to look for movies that are like Forrest Gump and Toy Story.
# To do this we need to find a Correlation between the two movies' ratings and other movies.
# This Correlates each movie to Forrest Gump
# Movies that have a rating closer to 1 indicates a very strong similarity between both movies
like_Forrest_Gump = matrix.corrwith(Forrest_Gump_rating)
like_Forrest_Gump.head()

# Use pandas corrwith to get the correlation between two dataframes.
# This Correlates each movie to Toy Story
# Movies that have a rating closer to 1 indicates a very strong similarity between both movies
like_Toy_Story = matrix.corrwith(Toy_Story_rating)
like_Toy_Story.head()

# In order of movies with the Strongest similarity to Least similarity
# The movie "The (Klass) Class (2007)" not only has a correlation of 1, but it is
# at the top of the list, which shows us that it is the most recommended movie.
correlation_Forrest_Gump = like_Forrest_Gump.sort_values(ascending=False)

# all recommendations with Very Strong similarity to Forrest Gump
# allRec = like_Forrest_Gump[like_Forrest_Gump == 1].index.tolist()
# print('\nList of all movies similiar to Forrest Gump: ', allRec)

# Top Five Recommendatins with the Strongest similiarity to Forrest Gump
topFiveRec = like_Forrest_Gump.sort_values(ascending=False).head(5)
print('\nTop 5 Recommendations similiar to Forrest Gump: ', topFiveRec)

# Most Recommended Movie with the Strongest similiarity to Forrest Gump
top_Forrest = like_Forrest_Gump.sort_values(ascending=False).head(1)
print('\nTop Recommendation similiar to Forrest Gump: ', top_Forrest)

