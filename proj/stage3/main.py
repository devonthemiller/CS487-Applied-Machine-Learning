import pandas as pd 
import numpy as np

# names = [userId#, movieId#, rating, timestamp]
ratings = pd.read_csv('ratings.csv')
ratings.head()

# names = [movieId#, title, genres]
movies = pd.read_csv('movies.csv')
movies.head()

# merge the ratings.csv and movies.csv on similiar "movieId"
dataset = pd.merge(ratings, movies, on='movieId')
dataset.head()
dataset.describe()

# userId - The ID of the user who rated the movie.
# movieId - The ID of the movie.
# rating - The rating the user gave the movie, between 0.5 and 5.0
# timestamp - The time the movie was rated, starting midnight (UTC) of January 1, 1970.
# title - The title of the movie.
# genres - The type of movie

# (1) Get Average Rating and (2) Number of Ratings FOR EACH Movie
avgRatings = pd.DataFrame(dataset.groupby('title')['rating'].mean())  # (1)
avgRatings.head()

# (3) Both Ratings will be used to calculate correlation between movies
# High Correlation = movies that are most similar, between -1 and 1.1
avgRatings['numRatings'] = dataset.groupby('title')['rating'].count() # (2)
avgRatings.head()

# visulize the distribution of the Ratings (see Plots tab)
avgRatings['rating'].hist(bins=50)  # most people chose 3, 3.5 or 4 (see peak)
                                    # and most ratings were between 3 and 4 (see spread)

# visualize the distribution of the Number of Ratings
avgRatings['numRatings'].hist(bins=60) # popular movies have most ratings, (see peak)
                                       # but most movies have few ratings (see spread)

# Check Relationship between Average Rating for a Movie and the Number of Ratings it recieved.
import seaborn as sns
sns.jointplot(x='rating', y='numRatings', data=avgRatings)
