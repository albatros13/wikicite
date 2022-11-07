import re
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from collections import Counter
from gensim.models import FastText
from sklearn import preprocessing
import json
import glob
import matplotlib.pyplot as plt
import gc

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import keras
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional

import tensorflow as tf
from tensorflow.python.client import device_lib

gc.collect()
warnings.filterwarnings("ignore")
tqdm.pandas()
keras.backend.backend()

local_device_protos = device_lib.list_local_devices()
print([x.name for x in local_device_protos if x.device_type == 'GPU'])
np.random.seed(0)

ext = "_xh"
PROJECT_HOME = 'c:///users/natal/PycharmProjects/cite-classifications-wiki/'

BOOK_JOURNAL_CITATIONS = PROJECT_HOME + 'data/features/book_journal_citations' + ext + '.parquet'
NEWSPAPER_CITATIONS = PROJECT_HOME + 'data/features/newspaper_citations' + ext + '.parquet'
ENTERTAINMENT_CITATIONS = PROJECT_HOME + 'data/features/entertainment_citations.parquet'
LARGEST_SECTIONS = PROJECT_HOME + 'data/features/largest_sections.csv'

TAG_COUNT = PROJECT_HOME +'/data/features/tag_counts.csv'

# from pyspark.sql import SparkSession
# spark = SparkSession.builder.appName("wiki qss").getOrCreate()

# Get auxiliary features and divide them into labels
# 1. `ref_index`
# 2. `total_words`
# 3. `tags`
# 4. `type_of_citation`
# Can include `section` of the page in which the citation belongs to

# BOOKS AND JOURNALS

book_journal_features = pd.read_parquet(BOOK_JOURNAL_CITATIONS, engine='pyarrow')

labels = ['doi', 'isbn', 'pmc', 'pmid', 'url', 'work', 'newspaper', 'website']
for label in labels:
    book_journal_features['citations'] = book_journal_features['citations'].progress_apply(
        lambda x: re.sub(label + '\s{0,10}=\s{0,10}([^|]+)', label + ' = ', x))

book_journal_features['actual_label'].value_counts()

journal_features = book_journal_features[book_journal_features['actual_label'] == 'journal']
book_features = book_journal_features[book_journal_features['actual_label'] == 'book']

# NEWSPAPERS

newspaper_data = pd.read_parquet(NEWSPAPER_CITATIONS, engine='pyarrow')

print('The total number of newspapers: {}'.format(newspaper_data.shape))

newspaper_data.head(2)
newspaper_data = newspaper_data[[
    'citations', 'ref_index', 'total_words', 'neighboring_words',
    'neighboring_tags', 'id', 'sections', 'type_of_citation'
]]
newspaper_data['actual_label'] = 'web'

newspaper_data.iloc[819218]['citations'] # Example before removing the fields
for label in labels:
    newspaper_data['citations'] = newspaper_data['citations'].progress_apply(
        lambda x: re.sub(label + '\s{0,10}=\s{0,10}([^|]+)', label + ' = ', x))

newspaper_data.iloc[819218]['citations'] # Example after removing the fields
newspaper_data = newspaper_data.sample(n=550000)

# entertainment_features = pd.read_parquet(ENTERTAINMENT_CITATIONS, engine='pyarrow')
# entertainment_features = entertainment_features[[
#     'ref_index', 'total_words', 'neighboring_words', 'neighboring_tags', 'id', 'sections', 'citations']]
#
# entertainment_features.iloc[23787]['citations'] # Example before removing the fields
#
# entertainment_features['actual_label'] = 'web'


# for label in labels:
#     entertainment_features['citations'] = entertainment_features['citations'].progress_apply(
#         lambda x: re.sub(label + '\s{0,10}=\s{0,10}([^|]+)', label + ' = ', x))
#
# entertainment_features.iloc[23787]['citations'] # Example after removing the fields
# entertainment_features['citations'].progress_apply(
#     lambda x: re.findall('{{\s{0,10}([^|]+)', x)[0].strip()).value_counts()

# entertainment_features = entertainment_features.sample(n=550000)


newspaper_data.drop('type_of_citation', axis=1, inplace=True)
book_journal_features.drop('type_of_citation', axis=1, inplace=True)

dataset_with_features = pd.concat([journal_features, book_features, newspaper_data])
    # , entertainment_features])
dataset_with_features.shape

le = preprocessing.LabelEncoder()
le.fit(dataset_with_features['actual_label'])
dataset_with_features['label_category'] = le.transform(dataset_with_features['actual_label'])
# dataset_with_features[dataset_with_features['actual_label'] == 'entertainment'].head(1)
dataset_with_features[dataset_with_features['actual_label'] == 'web'].head(1)
dataset_with_features[dataset_with_features['actual_label'] == 'book'].head(1)
dataset_with_features[dataset_with_features['actual_label'] == 'journal'].head(1)

# Convert citations' text to UTF-8
# NK We dont need that anymore
# dataset_with_features['citations'] = dataset_with_features['citations'].progress_apply(lambda x: x.encode("utf-8"))
print(dataset_with_features['actual_label'].value_counts())

# Clearing up memory
del book_journal_features
del newspaper_data
# del entertainment_features

# Remove rows which have duplicate ID and citations since they are just the same examples
dataset_with_features = dataset_with_features.drop_duplicates(subset=['id', 'citations']) ## keeps first row
dataset_with_features = dataset_with_features.reset_index(drop=True)
dataset_with_features.shape

# Save this file and use it - as an intermediate file if you want to use it somewhere else
dataset_with_features.to_csv('dataset_with_features.csv', index=False)

# Taking the unique `sections` and one hot encoding it to get a vector

# Only processing auxiliary features which are going to be used in the neural network
auxiliary_features = dataset_with_features[
    ['sections', 'citations', 'id', 'ref_index',
     'total_words', 'neighboring_tags', 'label_category']]


# auxiliary_features['sections'] = auxiliary_features['sections'].apply(
#     lambda x: x.encode('utf-8') if isinstance(x, unicode) else str(x))
auxiliary_features['sections'] = auxiliary_features['sections'].astype(str)
auxiliary_features['sections'] = auxiliary_features['sections'].apply(lambda x: x.split(', '))

section_counts = pd.Series(Counter(chain.from_iterable(x for x in auxiliary_features.sections)))
largest_sections = section_counts.nlargest(150)

largest_sections.to_csv(LARGEST_SECTIONS, header=None)

# Change section to `OTHERS` if occurence of the section is not in the 150 largest sections
auxiliary_features['sections'] = auxiliary_features['sections'].progress_apply(
    lambda x: list(set(['Others' if i not in largest_sections else i for i in x]))
)

print(auxiliary_features.head())

section_dummies = pd.get_dummies(auxiliary_features.sections.apply(pd.Series).stack())

auxiliary_features = auxiliary_features.join(section_dummies.sum(level=0))
auxiliary_features.drop('sections', axis=1, inplace=True)
auxiliary_features.head()

# Taking the `type of citations` and one hot encoding it to get a vector

# Get one hot encoding of citation_type column
# citation_type_encoding = pd.get_dummies(auxiliary_features['citation_type'])

# Drop column citation_type as it is now encoded and join it
# auxiliary_features = auxiliary_features.drop('citation_type', axis=1)

# Concat columns of the dummies along the axis with the matching index
# auxiliary_features = pd.concat([auxiliary_features, citation_type_encoding], axis=1)
# auxiliary_features.head()

# As we can see for the feature `total_number_of_words`, the mean and median **(since it is more robust in nature!)**
# are pretty high for articles which are `not` journal or books

print('Total mean length of entertainment articles: {}'.format( # Journal - length is less
    auxiliary_features[auxiliary_features['label_category'] == 1]['total_words'].mean()))
print('Total median length of entertainment articles: {}'.format(
    auxiliary_features[auxiliary_features['label_category'] == 1]['total_words'].median()))

print('Total mean length of journal articles: {}'.format(
    # Rest of the article have larger length
    auxiliary_features[auxiliary_features['label_category'] == 2]['total_words'].mean()))
print('Total median length of journal articles: {}'.format(
    auxiliary_features[auxiliary_features['label_category'] == 2]['total_words'].median()))

print('Total mean length of book articles: {}'.format( # Books - length is less
    auxiliary_features[auxiliary_features['label_category'] == 0]['total_words'].mean()))
print('Total median length of book articles: {}'.format(
    auxiliary_features[auxiliary_features['label_category'] == 0]['total_words'].median()))

# Taking the `neighboring_tags` and making an encoder dictionary for it
# To have more info about how what tag mean what:
# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

citation_tag_features = dataset_with_features[['id', 'citations', 'neighboring_tags']]

# citation_tag_features['neighboring_tags'] = citation_tag_features['neighboring_tags'].progress_apply(
#     lambda x: x.replace("'", "").replace('[', '').replace(']', '').replace('\n', '').split(' ')
# )

citation_tag_features.iloc[1]['neighboring_tags'][:10]

# Get the count for each POS tag so that we have an estimation as to how many are there
tag_counts = pd.Series(Counter(chain.from_iterable(x for x in citation_tag_features.neighboring_tags)))

# Considering the 10 smallest tags and checking which one does not have resemblance
tag_counts.nsmallest(10) 

tag_counts.to_csv(TAG_COUNT, header=None)

# We are going to replace `LS`, `the 2 backquotes` and the `the dollar symbol` since they do not have too much use case
# and do not give too much information about the context of the neighboring citation text.

OTHER_TAGS = ['LS', '``', '$']
citation_tag_features['neighboring_tags'] = citation_tag_features['neighboring_tags'].progress_apply(
    lambda x: [i if i not in OTHER_TAGS else 'Others' for i in x]
)

# Now, we can use the `count vectorizer` to represent the `POS tags` as a vector where each element of the vector
# represents the count of that tag in that particular citation.

cv = CountVectorizer() # Instantiate the vectorizer
citation_tag_features['neighboring_tags'] = citation_tag_features['neighboring_tags'].progress_apply(
    lambda x: " ".join(x))

transformed_neighboring_tags = cv.fit_transform(citation_tag_features['neighboring_tags'])
transformed_neighboring_tags = pd.DataFrame(transformed_neighboring_tags.toarray(), columns=cv.get_feature_names())

citation_tag_features.head()
transformed_neighboring_tags.shape, citation_tag_features.shape

citation_tag_features = citation_tag_features.reset_index(drop=True)
citation_tag_features = pd.concat([citation_tag_features, transformed_neighboring_tags], axis=1)
citation_tag_features.drop('neighboring_tags', axis=1, inplace=True)
citation_tag_features.head()

# Features for the LSTM - more time sequence related

# Citation's original text features

# Create a separate dataframe for preprocessing citation text
citation_text_features = dataset_with_features[['id', 'citations', 'label_category']]

# Convert the citation into a list by breaking it down into characters
citation_text_features['characters'] = citation_text_features['citations'].progress_apply(lambda x: list(x))

# Get the character counts for each unique character
char_counts = pd.Series(Counter(chain.from_iterable(x for x in citation_text_features.characters)))
char_counts.index

print('The max length of the longest citation in terms of characters is: {}'.format(
    max(citation_text_features.characters.apply(lambda x: len(x)))))

print('The mean length of the longest citation in terms of characters is: {}'.format(
    citation_text_features.characters.apply(lambda x: len(x)).mean()))

print('The median length of the longest citation in terms of characters is: {}'.format(
    citation_text_features.characters.apply(lambda x: len(x)).median()))

# Make a dictionary for creating a mapping between the char and the corresponding index
char2ind = {char: i for i, char in enumerate(char_counts.index)}
ind2char = {i: char for i, char in enumerate(char_counts.index)}

# Map each character into the citation to its corresponding index and store it in a list
X_char = []
for citation in citation_text_features.citations:
    citation_chars = []
    for character in citation:
        citation_chars.append(char2ind[character])
        
    X_char.append(citation_chars)

# Since the median length of the citation is 282, we have padded the input till 400 to get extra information which
# would be fed into the character embedding neural network.
# X_char = pad_sequences(X_char, maxlen=400)

# Append the citation character list with their corresponding lists for making a dataset
# for getting the character embeddings
data = []
for i in tqdm(range(len(X_char))):
    data.append((X_char[i], int(citation_text_features.iloc[i]['label_category'])))

# Separate out the training data and labels for further verification use
features = [i[0] for i in data]
labels = [i[1] for i in data]
# Changing it to dummy labels - identifier vs non identifier
labels = [i for i in labels]

# from collections import Counter
# Counter(labels) ## 1401521, 1651833

# Splitting the data into training and testing
training_data, testing_data, training_labels, testing_labels = train_test_split(
    features, labels, train_size=0.9, shuffle=True
)


# We are going to feed in the 400 character input since our median length comes out to be approximately 282 and train it
# on a dummy task - if the citation is scientific or not and get the embedding layer which would contain the
# representation for each character.

from keras.utils import to_categorical
categorical_labels = to_categorical(training_labels, num_classes=3)
categorical_test_labels = to_categorical(testing_labels, num_classes=3)


def citation_embedding_model():
    """
    Citation embedding generator model where the dimension of the embedding is 50.
    """
    main_input = Input(shape=(400, ), name='characters')
    # input dim is basically the vocab size
    emb = Embedding(input_dim=95, output_dim = 300, name='citation_embedding')(main_input)
    rnn = Bidirectional(LSTM(20))
    x = rnn(emb)
    de = Dense(3, activation='softmax')(x)
    model = Model(inputs = main_input, outputs = de)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Instantiate the model and generate the summary
model = citation_embedding_model()

model.summary()


def generator(features, labels, batch_size):
    """
    Generator to create batches of data so that processing is easy.
    
    :param: features: the features of the model.
    :param: labels: the labels of the model.
    :param: batch_size: the size of the batch
    """
    # Create empty arrays to contain batch of features and labels
    batch_features = np.zeros((batch_size, 400))
    batch_labels = np.zeros((batch_size, 3))
    while True:
        for i in range(batch_size):
            # choose random index in features
            index = np.random.choice(len(features), 1)[0]
            batch_features[i] = features[index]
            batch_labels[i] = categorical_labels[index]
        yield batch_features, batch_labels


# In[81]:


# Run the model with the data being generated by the generator with a batch size of 64
# and number of epochs to be set to 15
# hist = model.fit_generator(
#     generator(training_data, categorical_labels, 512), steps_per_epoch=4000, nb_epoch=2)


# Evaluation of embedding model
# y_predicted_proba = model.predict(np.array(testing_data))
# predicted_class = np.argmax(y_predicted_proba, axis=1)
# accuracy_score(testing_labels, predicted_class)

# Save the model so that we can retrieve it later
# model.save('/dlabdata1/harshdee/embedding_model.h5')
from keras.models import load_model
model = load_model('/dlabdata1/harshdee/embedding_model.h5')

# Get the `citation_embedding` layer and get the weights for each character
citation_layer = model.get_layer('citation_embedding')
citation_weights = citation_layer.get_weights()[0]
citation_weights.shape

# An example of the first element of an embedding
citation_weights[0][:100]


# In[86]:


# Map the embedding of each character to the character in each corresponding citation and aggregate (sum)
citation_text_features['embedding'] = citation_text_features['characters'].progress_apply(
    lambda x: sum([citation_weights[char2ind[c]] for c in x])
)


# In[87]:


# Normalize the citation embeddings so that we can check for their similarity later
citation_text_features['embedding'] = citation_text_features['embedding'].progress_apply(
    lambda x: x/ np.linalg.norm(x, axis=0).reshape((-1, 1))
)


# In[88]:


# Make the sum of the embedding to be summed up to 1
np.sum(np.square(citation_text_features['embedding'].iloc[0]))


# ### Similarity Graph for citation text embeddings

# In[89]:


# Just considering 20 since otherwise it will be computationally extensive
# citation_text_and_embeddings = citation_text_features[['citation', 'embedding']][:500]


# In[90]:


# citation_text_and_embeddings['embedding'] = citation_text_and_embeddings['embedding'].progress_apply(
#     lambda x: x[0].tolist()
# )


# In[91]:


# def tsne_embedding_plot():
#     labels = []
#     tokens = []

#     index = 0
#     for row in citation_text_and_embeddings:
#         tokens.append(row['embedding'])
#         labels.append(str(index))
#         index += 1
    
#     # Perplexity takes into account the global and local features
#     # We are using dimensionality reduciton for 2 features and taking 2500 iterations into account
#     tsne_model = TSNE(perplexity=40, n_components=2, n_iter=2500, random_state=0)
#     new_values = tsne_model.fit_transform(tokens)

#     x = []
#     y = []
#     for value in new_values:
#         x.append(value[0])
#         y.append(value[1])
        
#     plt.figure(figsize=(10, 10)) 
#     for i in range(len(x)):
#         plt.scatter(x[i],y[i])
#         plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2),
#                      textcoords='offset points', ha='right', va='bottom')
#     plt.show()


# In[92]:


# tsne_embedding_plot()


# In[93]:


# an example of citation embeddings which is close to each other
# citation_text_and_embeddings[citation_text_and_embeddings.index.isin([14, 477])] # (51, 243), (0, 13)


# In[94]:


# # Similiarity of 2 citations which are very similar
# result_similar = 1 - spatial.distance.cosine(
#     citation_text_and_embeddings.iloc[14]['embedding'],
#     citation_text_and_embeddings.iloc[477]['embedding']
# )
# result_similar


# In[95]:


# an example of citation embeddings which is NOT close to each other and are different
# citation_text_and_embeddings[citation_text_and_embeddings.index.isin([42, 124])] # (6, 42)


# In[96]:


# Similiarity of 2 citations which are not similar
# result_different = 1 - spatial.distance.cosine(
#     citation_text_and_embeddings.iloc[42]['embedding'],
#     citation_text_and_embeddings.iloc[124]['embedding']
# )
# result_different


# ### FastText embeddings for neighboring words

# In[97]:


# Load the pretrained embedding model on wikipedia
model = FastText.load_fasttext_format('/dlabdata1/harshdee/wiki.en.bin')


# In[98]:


# Create a separate dataframe for preprocessing citation words
citation_word_features = dataset_with_features[['id', 'citations', 'neighboring_words', 'label_category']]


# In[99]:


# Lowercase all the neighboring words for each of the citations
citation_word_features['neighboring_words'] = citation_word_features['neighboring_words'].progress_apply(
    lambda x: [i.lower() for i in x]
)


# Get the total unique words with their respective counts in the total dataset. This is done in order to remove words which are of low frequency and will potentially act as noise to the model.

# In[100]:


word_counts = pd.Series(Counter(chain.from_iterable(x for x in citation_word_features.neighboring_words)))


# In[101]:


threshold = 4

x = len(word_counts)
y = len(word_counts[word_counts <= threshold])
print('Total words: {}\nTotal number of words whose occurence is less than 4: {}\nDifference: {}'.format(x, y, x-y))
words_less_than_threshold = word_counts[word_counts <= threshold]


# In[102]:


# Remove the words which have a count of less than 4 and replace them with the unique <UNK> symbol
citation_word_features['neighboring_words'] = citation_word_features['neighboring_words'].progress_apply(
    lambda x: [i if i not in words_less_than_threshold else '<UNK>' for i in x]
)


# In[103]:


# creating a mapping between word and index or vice versa
words = pd.Series(Counter(chain.from_iterable(x for x in citation_word_features.neighboring_words))).index
word2ind = {w: i for i, w in enumerate(words)}
ind2words = {i: w for i, w in enumerate(words)}


# In[104]:


word_embedding_matrix = np.zeros((len(word2ind), 300))
for w in tqdm(word2ind):
    index = word2ind[w]
    word_embedding_matrix[index] = model.wv[w]


# Once we have the word embedding for each word in the neighboring words, we sum the embeddings for each word together in neighboring words to get an embedding which represents the past 40 words.

# In[105]:


citation_word_features['words_embedding'] = citation_word_features['neighboring_words'].progress_apply(
    lambda x: sum([word_embedding_matrix[word2ind[w]] for w in x])
)


# Now we have the `citation_word_features` and `citation_tag_features`, so we can join them together to form `time_sequence_features` which would be fed later into the LSTM..

# In[106]:


# Join time sequence features with the citations dataset
time_sequence_features = pd.concat([citation_tag_features, citation_word_features.reset_index(drop=True)], keys=['id', 'citations'], axis=1)
time_sequence_features = time_sequence_features.loc[:, ~time_sequence_features.columns.duplicated()]


# In[107]:


print('Total number of samples in time features are: {}'.format(time_sequence_features.shape))


# In[108]:


# citation_text = auxiliary_features.iloc[:,0]
# auxiliary_features['citation_text'] = citation_text
# auxiliary_features.drop('citation', axis=1, inplace=True)
# auxiliary_features.rename({'citation_text': 'citation'}, axis=1, inplace=True)


# In[109]:


# Join auxiliary features with the citations dataset
citation_text_features.reset_index(drop=True, inplace=True)
auxiliary_features.reset_index(drop=True, inplace=True)

auxiliary_features = pd.concat([auxiliary_features, citation_text_features], keys=['id', 'citations'], axis=1)
auxiliary_features = pd.concat([auxiliary_features['citations'], auxiliary_features['id']], axis=1)
auxiliary_features = auxiliary_features.loc[:, ~auxiliary_features.columns.duplicated()]
auxiliary_features.shape


# In[110]:


# Drop columns with are duplicates
auxiliary_features.drop(['neighboring_tags', 'characters'], axis=1, inplace=True)


# In[111]:


del model
del word_embedding_matrix
del citation_word_features
del citation_text_features

gc.collect()


# ## Making sets for `auxiliary` and `time sequence` features

# In[112]:


data = dataset_with_features[['id', 'citations', 'label_category']]


# In[113]:


# Join the time sequence features for the data
time_sequence_features = pd.concat([time_sequence_features['id'], time_sequence_features['citations']], axis=1)
time_sequence_features = pd.concat([time_sequence_features, data.reset_index(drop=True)], keys=['id', 'citations'], axis=1)
time_sequence_features.columns = time_sequence_features.columns.droplevel(0)
time_sequence_features = time_sequence_features.loc[:, ~time_sequence_features.columns.duplicated()]


# In[114]:


time_sequence_features['words_embedding'] = time_sequence_features['words_embedding'].progress_apply(
    lambda x: x.tolist())


# In[115]:


auxiliary_features['embedding'] = auxiliary_features['embedding'].progress_apply(lambda x: x.tolist())


# In[116]:


len(time_sequence_features), len(auxiliary_features)


# In[117]:


del book_features
del journal_features
gc.collect()


# ## Splitting the dataset into training, testing and validation 

# The split is done into 80-10-10 ratio so that we have more training data to train on and have validation dataset to make sure that the model is working as anticipated.

# In[118]:


type(auxiliary_features)


# In[119]:


# Get the labels which will be split later
y = auxiliary_features.loc[:, 'label_category'].astype(int).tolist()


# In[120]:


# Make a mask for auxiliary dataset to get all features except the one below
column_mask_aux = ~auxiliary_features.columns.isin(['id', 'citations', 'label_category'])


# In[121]:


# # Get the columns of those auxiliary features and covert them into a list
auxiliary = auxiliary_features.loc[:, column_mask_aux].values.tolist()


# In[122]:


# # Convert them into numpy array (for Keras) and stack them (if needed) as suited for the model's format
auxiliary = [np.array(auxiliary[i][0][0] + auxiliary[i][1:]) for i in tqdm(range(len(auxiliary)))]


# In[123]:


# # Make a mask for time sequences features dataset to get all features except the one below
cols = [col for col in time_sequence_features.columns if col not in ['id', 'citations', 'label_category', 'neighboring_words']]
stripped_tsf = time_sequence_features[cols]


time = stripped_tsf.values.tolist()


def make_structure_time_features(time_features):
    """
    Concatenate features which are numbers and lists together by checking the type:
    param: time_features: the features which are considered time sequence.
    """
    feature_one = np.array([int(i) for i in time_features if isinstance(i, int)])
    # NK replaced long to int
    feature_two = np.array([i for i in time_features if isinstance(i, list)][0])
    return np.array([feature_one, feature_two])


time = [make_structure_time_features(time[i]) for i in tqdm(range(len(time)))]

# Instantiating PCA to 35 components since it should be equal to the size of the vector of the tags
pca = PCA(n_components=35)


def get_reduced_words_dimension(data):
    """
    Get the aggregated dataset of words and tags which has the
    same dimensionality using PCA.
    
    :param: data: data which needs to be aggregated.
    """
    tags = [i for i, _ in data]
    word_embeddings = [j for _,j in data]
    pca.fit(word_embeddings)
    
    word_embeddings_pca = pca.transform(word_embeddings)
    tags = np.array(tags)
    return word_embeddings_pca, tags

# Apply PCA on all the sets of data to have the dimensions of the data to be the same
word_embeddings_pca, tags = get_reduced_words_dimension(time)


time_pca = np.dstack((word_embeddings_pca, tags))


word_embeddings_pca.shape, tags.shape, time_pca.shape


del time_sequence_features
del auxiliary_features


# In[132]:


# del data
del word_embeddings_pca
del tags
del stripped_tsf
del column_mask_aux
gc.collect()


# ## LSTM/Neural Network Model

# In[133]:


def generator_nn(features_aux, features_time, labels, batch_size):
    """
    Generator to create batches of data so that processing is easy.
    
    :param: features: the features of the model.
    :param: labels: the labels of the model.
    :param: batch_size: the size of the batch
    """
    # Create empty arrays to contain batch of features and labels
    batch_features_aux = np.zeros((batch_size, 453))
    batch_features_time =  np.zeros((batch_size, 35, 2))
    batch_labels = np.zeros((batch_size, 3))
    while True:
        for i in range(batch_size):
            # choose random index in features
            index = np.random.choice(len(features_aux), 1)[0]
            batch_features_aux[i] = features_aux[index]
            batch_features_time[i] = features_time[index]
            batch_labels[i] = labels[index]
        yield [batch_features_time, np.asarray(batch_features_aux)], batch_labels


# In[134]:


from keras.optimizers import Adam


# In[135]:


def scheduler(epoch, lr):
    import math
    if epoch < 10:
        return lr
    else:
        return lr * math.exp(-0.1)


# In[136]:


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


# In[137]:


def classification_model():
    """
    Model for classifying whether a citation is scientific or not.
    """
    main_input = Input(shape=(35, 2), name='time_input')
    lstm_out = LSTM(64)(main_input)

    auxiliary_input = Input(shape=(453,), name='aux_input') ## 454 without citation type, 476 with citation type
    # Converging the auxiliary input with the LSTM output
    x = keras.layers.concatenate([lstm_out, auxiliary_input])

    # 4 fully connected layer
    x = Dense(256, activation='selu')(x)
    x = Dense(128, activation='selu')(x)
    x = Dense(128, activation='selu')(x)
    x = Dense(64, activation='selu')(x)

    main_output = Dense(3, activation='softmax', name='main_output')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])
    
    opt = Adam(0.001)
    model.compile(
        optimizer=opt, loss={'main_output': 'categorical_crossentropy'},
        loss_weights={'main_output': 1.}, metrics=['acc']
    )
    return model


# In[138]:


# Instantiating the classification model
model = classification_model()
model.summary()


# We use `ReduceLRonPlateau` so that the model does not overshoot the optimal minimum point and hence by default we start with a learning rate of 0.01 but as soon as the accuracy stop increasing the learning rate does not change which helps us converge better.

# In[139]:


## Convert auxiliary into numpy array for indexing
auxiliary = np.asarray(auxiliary)
y = np.asarray(y)


# In[140]:


EPOCHS = 30


# In[141]:


x_train_indices, x_test_indices, y_train_indices, y_test_indices = train_test_split(
    range(auxiliary.shape[0]), range(y.shape[0]), train_size=0.9, stratify=y, shuffle=True
)


# In[142]:


aux_train = auxiliary[x_train_indices]
time_train = time_pca[x_train_indices]
y_train = np.eye(3)[y[x_train_indices]]


# In[143]:


aux_test = auxiliary[x_test_indices]
time_test = time_pca[x_test_indices]
y_test = y[x_test_indices]


# In[ ]:


# predictions = []
# for index, (train_indices, val_indices) in enumerate(skf.split(auxiliary, y)):
#     aux_train, aux_val = auxiliary[train_indices], auxiliary[val_indices]
#     time_train, time_val = time_pca[train_indices], time_pca[val_indices]
#     y_train = np.eye(4)[y[train_indices]]
#     y_val = y[val_indices]
    
BATCH_SIZE = 256
print('Running model with epochs: {}'.format(EPOCHS))

model = None
model = classification_model()
training_generator = generator_nn(aux_train, time_train, y_train, BATCH_SIZE)

history_callback = model.fit_generator(
    training_generator,
    steps_per_epoch=len(x_train_indices) // 256,
    epochs=EPOCHS, verbose=1, shuffle=True, callbacks=[callback]
)


# In[ ]:


# Running model with epochs: 5
# Epoch 1/5
# 11601/11601 [==============================] - 421s 36ms/step - loss: 0.7328 - acc: 0.7955
# Epoch 2/5
# 11601/11601 [==============================] - 413s 36ms/step - loss: 0.3311 - acc: 0.8823
# Epoch 3/5
# 11601/11601 [==============================] - 413s 36ms/step - loss: 0.2957 - acc: 0.8946
# Epoch 4/5
# 11601/11601 [==============================] - 406s 35ms/step - loss: 0.2777 - acc: 0.9024
# Epoch 5/5
# 11601/11601 [==============================] - 410s 35ms/step - loss: 0.2617 - acc: 0.9085
# ---------------------------------------------------------------------------


# In[ ]:


history_dict = history_callback.history


# In[ ]:


f = open('/dlabdata1/harshdee/results/citation_model_loss_{}.json'.format(EPOCHS), 'w')
f.write(str(history_dict))
f.close()

prediction_for_folds = model.predict([time_test, aux_test])
y_pred = np.argmax(prediction_for_folds, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the Neural network model for epochs {}: {}".format(EPOCHS, accuracy))

res = pd.DataFrame(confusion_matrix(y_test, y_pred))
res.index = ['book', 'journal', 'web']
res.columns = ['book', 'journal', 'web']
res['accuracy'] = accuracy
res.to_csv('/dlabdata1/harshdee/results/citation_model_result_{}.csv'.format(EPOCHS))
print(res)

model.save('/dlabdata1/harshdee/results/citation_model_epochs_{}.h5'.format(EPOCHS))
json_string = model.to_json()
with open("/dlabdata1/harshdee/results/citation_model_epochs_{}.json".format(EPOCHS), "w") as json_file:
    json_file.write(json_string)

print('\n\nDone with the prediction and saving model with epochs: {}\n'.format(EPOCHS))
