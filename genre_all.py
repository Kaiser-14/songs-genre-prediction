import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import SnowballStemmer
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from random import randint

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Import the dataset
print("Starting...")
# songs_df = pd.read_csv('./datasets/lyrics.csv') # Without processing
# songs_df = pd.read_csv('./datasets/lyrics_processed.csv')  # Dataset stemmed and taking only lyrics with at least 300 characters
songs_df = pd.read_csv('./datasets/lyrics_processed2.csv')  # Dataset stemmed and taking only lyrics with at least 100 characters
songs_df = songs_df.set_index('index')
songs_df = songs_df.dropna(inplace=False)
# print(songs_df.head()) # Checking initial information

# There are some rows in 'year' column that include text instead of integers. Although they have been deleted when
# dropping NaN rows, it is useful to do a complete processing.
songs_df = songs_df.loc[songs_df['year'].apply(type) == int]

# Some rows have some genres not useful for this practice (Not Available & Other). As I did not use for predictions,
# I can remove these rows.
# print(len(songs_df))
songs_df = songs_df.loc[songs_df['genre'] != 'Not Available']
songs_df = songs_df.loc[songs_df['genre'] != 'Other']
# print(len(songs_df))

# Checking visually into the dataset, I find that there are many lyrics line with wrong data. They correspond to some
# specific artist (maybe something went wrong when inserting them). Finally, take only songs with at least 100
# characters.
# print(len(songs_df))
songs_df = songs_df.loc[songs_df['artist']!= 'george-harrison']
songs_df = songs_df.loc[songs_df['artist']!= 'adaaeaaineay-iaidiia']
songs_df = songs_df.loc[songs_df['artist']!= 'ddoduddegd1-2-dd-nd-d']
songs_df = songs_df.loc[songs_df['artist']!= 'dicaiaaoi-aeaenaiad']
songs_df = songs_df.loc[songs_df['lyrics'].str.len() > 100]
# print(len(songs_df))

# Due to previous dropping, I should reset the index, and maintain the old index for detecting easily the songs
songs_df = songs_df.reset_index()
# Uncomment when using the complete dataset to run faster
#songs_df = songs_df.loc[:49999]
# print(len(songs_df))
# print(songs_df.head())

print("Number of different artists:", len(songs_df['artist'].unique()))
print("Number of genres:", len(songs_df['genre'].unique()))

# I have to preprocess a little bit the lyrics. Only stemming the words and removing stop words. Additionally, to
# improve the model, I lowercase all of them
stemmer = SnowballStemmer('english')
words = stopwords.words("english")
# Uncomment when using the complete dataset to run faster. That process takes too much time.
# print("Processing lyrics...")
# songs_df['lyrics'] = songs_df['lyrics'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x)
# .split() if i not in words]).lower())
# print(songs_df['lyrics'].head())
# export_csv = songs_df.to_csv (r'export_dataframe2.csv', index = None, header=True)

# Some genres can be combined to improve results and increment the number of items per genre. Also mention that some
# could be part of others, causing problems in the classification (e.g.: Indie can be included in Pop or Rock). After
# testing different approaches to combine genres, I decided to have only some of them, the most representative and the
# most differentiate possible between them. If Pop is included into the list of genres, I get 10% lower in the precision
# for predict songs, but it could be assumable as I have checked that it only mix data with Rock genre.
songs_df['genre'] = np.where((songs_df['genre'] == 'Folk'), 'Country', songs_df['genre'])
# songs_df['genre'] = np.where((songs_df['genre'] == 'Jazz'), 'Hip-Hop', songs_df['genre'])
# songs_df['genre'] = np.where((songs_df['genre'] == 'R&B'), 'Pop', songs_df['genre'])
# songs_df['genre'] = np.where((songs_df['genre'] == 'Electronic'), 'Metal', songs_df['genre'])
# songs_df['genre'] = np.where((songs_df['genre'] == 'Indie'), 'Rock', songs_df['genre'])
# genres = ['Country', 'Electronic', 'Folk', 'Hip-Hop', 'Indie', 'Jazz', 'Metal', 'Pop', 'R&B', 'Rock']
# genres = ['Country', 'Hip-Hop', 'Metal', 'Pop', 'Rock']
genres = ['Country', 'Electronic', 'Hip-Hop', 'Indie', 'Jazz', 'Metal', 'Pop', 'R&B', 'Rock']

# After cleaning the lyrics and the dataset, these are the values that I have for each genre (training and test values)
# COUNTRY 8945 3833
# ELECTRONIC 4283 1836
# FOLK 1315 563
# HIP-HOP 16651 7136
# INDIE 1885 808
# JAZZ 4555 1952
# METAL 13422 5752
# POP 26316 11279
# R&B 2161 926
# ROCK 65908 28247

# Next part is to divide the dataset in small samples for each genre. As I have too much songs for Rock, I had to
# take only a part of them
print("--------\nPreparing training/testing subsets for each genre...")
train_df = pd.DataFrame()
test_df = pd.DataFrame()
# As mentioned before, taking a small subsets for each genre. The most difficult part is to tune that values to have
# later good results
for genre in genres:
    subset = songs_df[(songs_df['genre'] == genre)]
    #if genre == 'Pop': subset = subset[:18000]
    #if genre == 'Metal': subset = subset[:18000]
    subset = subset[:15000]

    N = round(len(subset)*0.7)  # 70% training, 30% testing
    train_set = subset.sample(n=N)
    test_set = subset.drop(train_set.index)
    print("{0}. Training set of {1} songs. Test set of {2} songs".format(genre, len(train_set), len(test_set)))
    train_df = train_df.append(train_set)
    test_df = test_df.append(test_set)

print("--------\nCreating and training the model...")
# Defining the model
text_clf = Pipeline(
    [('vect', TfidfVectorizer(
        ngram_range=(1, 2),  # include bi-grams
        max_df=0.4,  # ignore terms that appear in more than 40% of documents
        min_df=4)),  # ignore terms that appear in less than 4 documents
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB(alpha=0.1))])

# Training the model
text_clf.fit(train_df['lyrics'], train_df['genre'])

# Let's take some scores based on the testing data. I receive a prediction for each lyric, and checking the total
# precision for the model
predicted = text_clf.predict(test_df['lyrics'])
print("Model precision:", np.mean(predicted == test_df['genre']))
# 73% for N=1000 and 90k dataset and 3 Genres (Indie, Jazz, Rock)
# 62.57% for N=100 and 10k dataset and 3 Genres (Indie, Jazz, Rock)
# 79% for the complete dataset and 2 genres (Jazz, Rock)

# In order to improve the tunning for taking the subsets, I created the confusion matrix that could provided insights
# in terms of model accuracy for each genre.
mat = confusion_matrix(test_df['genre'], predicted)
sns.heatmap(
    mat.T, square=True, annot=True, fmt='d', cbar=False,
    xticklabels=genres,
    yticklabels=genres
)
plt.xlabel('Correct label')
plt.ylabel('Predicted label')
plt.title('Correlation Matrix - All genres')
plt.show()

# After some counts numbers, it is useful some extra data for checking results. In this case, 4 new metrics to compare
# results.
print("Other metrics.")
precision, recall, fscore, support = precision_recall_fscore_support(test_df['genre'], predicted)

for n, genre in enumerate(genres):
    genre = genre.upper()
    print(genre+'_precision: {}'.format(precision[n]))  # Precision is the ability to not label a negative result as a
    # positive one.
    print(genre+'_recall: {}'.format(recall[n]))  # Recall is the percentage in finding all the positive results.
    print(genre+'_fscore: {}'.format(fscore[n]))  # Harmonic mean between precision and recall.
    print(genre+'_support: {}'.format(support[n]))  # Number of songs tested.

# I have a model and trained with many songs, so now I have to predict some lyrics. As I mentioned in the Summary
# Proposal, a sentence could be provided and the model will show the genre most similar to it. It works the same as the
# lyrics of a song, which has several sentences (and not every of them are predicted for the same genre).
lyric = "Let me be inspired And let me love Nothing for me You can't be the one Smell of betrayal I love to hide " \
          "Today i "
lyric_proc = " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", lyric).split() if i not in words]).lower()

predicted = text_clf.predict([lyric_proc])
probabilities = text_clf.predict_proba([lyric_proc])
print("--------\nPrediction for the inserted sentence")
print("Predicted genre:", predicted[0])
print("Probability of being part of each genre is:")
print("Country:", probabilities[0][0])
print("Electronic:", probabilities[0][1])
print("Hip-Hop:", probabilities[0][2])
print("Indie:", probabilities[0][3])
print("Jazz:", probabilities[0][4])
print("Metal:", probabilities[0][5])
print("Pop:", probabilities[0][6])
print("R&B:", probabilities[0][7])
print("Rock:", probabilities[0][8])

# Let's test the data with a similar music database and see what results I obtain. In this case, the dataset has songs,
# artist and lyrics columns (more columns but not interesting for this case), but not a genre column. I predict the
# genre thanks to my model, and returned the probabilities for be part of any genre. The genre predicted will be the
# higher value. Finally, it is provided a list of 10 songs of the predicted genre. Note that I have commented the first
# lines in the same way as the first lines of the program, due to the delay to process that second dataset. I exported
# it so now I can import it in my code easily. Also mention that the song that I am going to predict is selected
# randomly, based on the index of that second dataset.
# test_songs_df = pd.read_csv('./datasets/songdata.csv')
# test_songs_df = test_songs_df.dropna(inplace = False)
# test_songs_df['text'] = test_songs_df['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]",
# " ", x).split() if i not in words]).lower())
# export_csv = test_songs_df.to_csv (r'export_dataframe2.csv', index = None, header=True)
# I have
# preprocessed the document and save it in a file
test_songs_df = pd.read_csv('./datasets/songdata_processed.csv')  # Second dataset, with no genre columns and lyrics processed
index_song = randint(0, len(test_songs_df)-1)
lyric = test_songs_df['text'].iloc[index_song]

predicted = text_clf.predict([lyric])
probabilities = text_clf.predict_proba([lyric])
song = test_songs_df['song'].iloc[index_song]
artist = test_songs_df['artist'].iloc[index_song]
song = song.replace(' ', '-')
song = song.replace("'", '')
artist = artist.replace(' ', '-')
artist = artist.replace("'", '')
link = "https://www.google.es/search?q=" + song + "+" + artist + "+" + "genre"  # Search for genre in Google
print("--------\nPrediction for song:", test_songs_df['song'].iloc[index_song])
print("Predicted genre:", predicted[0])
print("Probability of being part of each genre is:")
print("Country:", probabilities[0][0])
print("Electronic:", probabilities[0][1])
print("Hip-Hop:", probabilities[0][2])
print("Indie:", probabilities[0][3])
print("Jazz:", probabilities[0][4])
print("Metal:", probabilities[0][5])
print("Pop:", probabilities[0][6])
print("R&B:", probabilities[0][7])
print("Rock:", probabilities[0][8])
print("Check the genre in Google:", link)

# Return 10 songs similar of the same genre
rnd_subset = songs_df.sample(n=10)
print("--------\nHere a list of 10 songs for genre:", predicted[0])
for song, artist, loop in zip(rnd_subset['song'], rnd_subset['artist'], range(1, 11)):
    link = "https://music.youtube.com/search?q="+ song + "-" + artist
    print("{0} {1}. By {2} {3}".format(loop, song.replace('-', ' ').capitalize(), artist.replace('-', ' ').capitalize(),
                                       link))

# Now the idea is to create a 2D map, to check visually the different genres and see the results in the map. I use PCA
# and t-SNE to reduce the dimensions, but first, I need to score every song in the initial dataset into the 5 main
# genres taken for this practice. Later, I included the extracted song of the second dataset into the 2D map and see
# what happens (and see if it corresponds to the predicted genre).

# If I take the whole dataset, both PCA and t-SNE provided bad looking scatter, so I took only 1000 entries and see
# possible results. Note that I take 1000 entries for the specific genres (genres used for the model).
# Select one of the two options: taking the whole dataset for the specific genres or take only 1000 entries of the
# complete dataset, including another genres. To be able to compare with some genres, I need to predict the whole
# dataset, adding extra columns corresponding to the predictions probabilities for be part of each genre. In the same
# way as previous sections, I exported the data frame into a file to import it easily.
# map_songs=train_df.append(test_df)
# map_songs = songs_df.sample(999)
# probabilities = text_clf.predict_proba(map_songs['lyrics'])  # Predict for the lyrics provided
#
# map_songs['Country'], map_songs['Electronic'], map_songs['Hip-Hop'], map_songs['Indie'], map_songs['Jazz'], \
#     map_songs['Metal'], map_songs['Pop'], map_songs['R&B'], map_songs['Rock'] = \
#     ['', '', '', '', '', '', '', '', '']
# for n,row in enumerate(map_songs.itertuples()):  # Merge scored data into the dataset
#    map_songs.loc[row.Index, 'Country'] = probabilities[n][0]
#    map_songs.loc[row.Index, 'Electronic'] = probabilities[n][1]
#    map_songs.loc[row.Index, 'Hip-Hop'] = probabilities[n][2]
#    map_songs.loc[row.Index, 'Indie'] = probabilities[n][3]
#    map_songs.loc[row.Index, 'Jazz'] = probabilities[n][4]
#    map_songs.loc[row.Index, 'Metal'] = probabilities[n][5]
#    map_songs.loc[row.Index, 'Pop'] = probabilities[n][6]
#    map_songs.loc[row.Index, 'R&B'] = probabilities[n][7]
#    map_songs.loc[row.Index, 'Rock'] = probabilities[n][8]
#
# print("Done")
# export_csv = map_songs.to_csv (r'export_dataframe.csv', index = None, header=True)
# columns_of_interest = ['artist', 'song', 'genre', 'Country', 'Hip-Hop', 'Metal', 'Pop', 'Rock']
# print(map_songs[columns_of_interest].head())

# t-SNE
# The process to predict the complete dataset is very slow, so as I mentioned, I exported the file to then have the
# possibility to save time. Also, after some iterations, I discovered that the best way to handle that last exercise
# was to take 1000 values from the complete dataset (including all genres) and the then make the t-SNE for the specific
# genres. Finally, plot the t-SNE results and later include the results for the previous random predicted song.
map_songs = pd.read_csv('./datasets/final_small_all.csv')  # 1000 entries with all the genres, lyrics processed and added probabilities
# map_songs = pd.read_csv('./datasets/final_genres_all.csv')  # 10000 entries with all the genres, lyrics processed and added probabilities
RANDOM_STATE = 42

# features = ['Country', 'Hip-Hop', 'Metal', 'Pop', 'Rock']
features = ['Country', 'Electronic', 'Hip-Hop', 'Indie', 'Jazz', 'Metal', 'Pop', 'R&B', 'Rock']
data_subset = map_songs[features].values

# Adding the prediction probabilities for the previous song
# probabilities_df = pd.DataFrame(data=probabilities, columns=['Country', 'Hip-Hop', 'Metal', 'Pop', 'Rock'])
probabilities_df = pd.DataFrame(data=probabilities, columns=['Country', 'Electronic', 'Hip-Hop', 'Indie', 'Jazz',
                                                             'Metal', 'Pop', 'R&B', 'Rock'])
map_features = map_songs[features]
map_features = map_features.append(probabilities_df, ignore_index=True)

# TSNE
tsne = TSNE(n_components=2, n_iter=250, random_state=RANDOM_STATE)
points = tsne.fit_transform(map_features)
result_TSNE = pd.DataFrame(data=points, columns=['TSNE 1', 'TSNE 2'])
result = pd.concat([result_TSNE, map_songs[['genre']]], axis=1)  # Concatenate genres to the results
result.iloc[result.shape[0]-1, 2] = 'Song'  # Needed to identify the random songs
# export_csv = result.to_csv(r'test.csv', index=None, header=True)

# Showing the scatter
# targets = ['Country', 'Hip-Hop', 'Metal', 'Pop', 'Rock', 'Song']
targets = ['Country', 'Electronic', 'Hip-Hop', 'Indie', 'Jazz', 'Metal', 'Pop', 'R&B', 'Rock', 'Song']
colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'brown', 'purple', 'pink', 'black']

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('TSNE 1', fontsize=15)
ax.set_ylabel('TSNE 2', fontsize=15)
ax.set_title('t-SNE', fontsize=20)
for target, color in zip(targets, colors):
    indicesToKeep = result['genre'] == target
    if target != 'Song':
        ax.scatter(result.loc[indicesToKeep, 'TSNE 1']
                   , result.loc[indicesToKeep, 'TSNE 2']
                   , c=color
                   , s=30)
    if target == 'Song':
        ax.scatter(result.loc[indicesToKeep, 'TSNE 1']
                   , result.loc[indicesToKeep, 'TSNE 2'], 100.0, marker='X', c=color)
ax.legend(targets)
ax.grid()
plt.show()
