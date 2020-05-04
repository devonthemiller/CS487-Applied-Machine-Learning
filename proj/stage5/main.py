import pandas as pd 
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

#----------------------------------------------------------------------
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
#----------------------------------------------------------------------
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
Pulp_Fiction_rating = matrix['Pulp Fiction (1994)']
Jurrassic_Park_rating = matrix['Jurassic Park (1993)']
Braveheart_rating = matrix['Braveheart (1995)']
Schindlers_List_rating = matrix["Schindler's List (1993)"] 
# Terminator_2_rating = matrix['Terminator 2: Judgement Day (1991)']
# The_Shawshank_Redemption_rating = matrix['The Shawshank Redemption (1994)']
# The_Silence_of_the_Lambs_rating = matrix['The Silence of the Lambs (1991)']
# A_New_Hope_Star_Wars_rating = matrix['A New Hope Star Wars: Episode IV (1977)']
# The_Matrix_rating = matrix['The Matrix (1999)']

Forrest_Gump_rating.head()
Pulp_Fiction_rating.head()

# Goal is to find movies that are similar Forrest Gump and Pulp Fiction.
# To do this we need to find a Correlation between the two movies' ratings and other movies.
# This Correlates each movie to Forrest Gump
# Movies that have a rating closer to 1 indicates a very strong similarity between both movies
like_Forrest_Gump = matrix.corrwith(Forrest_Gump_rating)
like_Forrest_Gump.head()

# Use pandas corrwith to get the correlation between two dataframes.
# This Correlates each movie to Pulp Fiction
# Movies that have a rating closer to 1 indicates a very strong similarity between both movies
like_Pulp_Fiction = matrix.corrwith(Pulp_Fiction_rating)
like_Pulp_Fiction.head()
#-----------------------------------------------------------------------------
# In order of movies with the Strongest similarity to Least similarity
# The movie "The (Klass) Class (2007)" not only has a correlation of 1, but it is
# at the top of the list, which shows us that it is the most recommended movie.
correlation_Forrest_Gump = like_Forrest_Gump.sort_values(ascending=False)

# Coffee Town (2013) is the most recommended movies
correlation_Pulp_Fiction = like_Pulp_Fiction.sort_values(ascending=False)

# all recommendations with Very Strong similarity to Forrest Gump
# allRec = like_Forrest_Gump[like_Forrest_Gump == 1].index.tolist()
# print('\nList of all movies similiar to Forrest Gump: ', allRec)

# Top Five Recommendatins with the Strongest similiarity to Forrest Gump
topFive_Forrest = like_Forrest_Gump.sort_values(ascending=False).head(5)
# print('\nTop 5 Recommendations similiar to Forrest Gump: \n', topFive_Forrest)

# Most Recommended Movie with the Strongest similiarity to Forrest Gump
top_Forrest = like_Forrest_Gump.sort_values(ascending=False).head(1)
# print('\nTop Recommendation similiar to Forrest Gump: \n', top_Forrest)

# Top Five Recommendatins with the Strongest similiarity to Pulp Fiction
topFive_Pulp_Fiction = like_Pulp_Fiction.sort_values(ascending=False).head(5)
# print('\nTop 5 Recommendations similiar to Pulp Fiction: \n', topFive_Pulp_Fiction)

# Most Recommended Movie with the Strongest similiarity to Pulp Fiction
top_Pulp = like_Pulp_Fiction.sort_values(ascending=False).head(1)
# print('\nTop Recommendation similiar to Pulp Fiction: \n', top_Pulp)
#-----------------------------------------------------------------------------
# print( correlation_Forrest_Gump.head() )

# If user didn't rate the movie; missing values; drop null values
corr_Forrest = pd.DataFrame(correlation_Forrest_Gump, columns=['Correlation'])
corr_Forrest.dropna(inplace=True)
# print( corr_Forrest.head() )
# produced same result

# print( correlation_Pulp_Fiction.head() )

# If user didn't rate the movie; missing values; drop null values
corr_pulp = pd.DataFrame(correlation_Pulp_Fiction, columns=['Correlation'])
corr_pulp.dropna(inplace=True)
# print( corr_pulp.head() )
# produced same output
#-----------------------------------------------------------------------------
# Handle movies with few ratings, but High ratings.
# Fix this problem by setting Threshold for the number of ratings.
# Histogram shows that there is a sharp decline in number of ratings from 100
# To do this, join the two dataframes with the total_ratings column
# in the ratings DataFrame
corr_Forrest = corr_Forrest.join(ratings['total_ratings'])
corr_pulp = corr_pulp.join(ratings['total_ratings'])

corr_Forrest .head()
corr_pulp.head()

# Find movies that are most similar to title with atleast 100 reviews
# then sort by Correlation column and view first 4 rows
top3_Forrest = corr_Forrest[corr_Forrest['total_ratings'] > 100].sort_values(by='Correlation', 
                                                              ascending=False).head(4)

top3_Pulp = corr_pulp[corr_pulp['total_ratings'] > 100].sort_values(by='Correlation', 
                                                        ascending=False).head(4)

# The first film in the printout will be of the movie itself, this
# shows a perfect correlation with itself. We will drop first row.

top3_Forrest = top3_Forrest.drop(['Forrest Gump (1994)'])
print('Top 3 most recommended movie out of 100,000 films most similar to Forrest Gump \n', top3_Forrest)
# The most similar movie to Forrest Gump would be in the next 
# row, which is "Good Will Hunting (1997), with a correlation of 0.54"

print('\n')

top3_Pulp = top3_Pulp.drop(['Pulp Fiction (1994)'])
print('Top 3 most recommended movie out of 100,000 films most similar to Pulp Fiction \n', top3_Pulp)
# The most similar movie to Pulp Fiction would be in the next 
# row, which is "Fight Club (1999), with a correlation of 0.54"







