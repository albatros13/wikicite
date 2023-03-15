import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer


def make_structure_time_features(time_features):
    """
    Concatenate features which are numbers and lists together by checking the type:
    param: time_features: the features which are considered time sequence.
    """
    # NK replaced long to int
    feature_one = np.array([i for i in time_features if isinstance(i, int)])
    feature_two = np.array([i for i in time_features if isinstance(i, list)][0])

    # NK fixing dimension
    if len(feature_two) == 1:
        feature_two = [0]*300
    return np.array([feature_one, feature_two])


def encode_auxuliary(aux_features):
    # Make a mask for auxiliary dataset to get all features except the one below
    column_mask_aux = ~aux_features.columns.isin(['id', 'citations', 'label_category', 'neighboring_words'])
    # Get the columns of those auxiliary features and covert them into a list
    auxiliary = aux_features.loc[:, column_mask_aux].values.tolist()
    # Convert them into numpy array (for Keras) and stack them (if needed) as suited for the model's format
    auxiliary = [np.array(auxiliary[i][0][0] + auxiliary[i][1:]) for i in tqdm(range(len(auxiliary)))]

    print("Aux features: ", aux_features.columns)

    return np.asarray(auxiliary)


def join_auxiliary_with_text(aux_features, text_features):
    # Join auxiliary features with the citations dataset
    text_features.reset_index(drop=True, inplace=True)
    aux_features.reset_index(drop=True, inplace=True)
    aux_features = pd.concat([aux_features, text_features], keys=['id', 'citations'], axis=1)
    aux_features = pd.concat([aux_features['citations'], aux_features['id']], axis=1)
    aux_features = aux_features.loc[:, ~aux_features.columns.duplicated()]
    aux_features['embedding'] = aux_features['embedding'].progress_apply(lambda x: [x] if isinstance(x, int) else x.tolist())
    aux_features.drop(['neighboring_tags', 'characters'], axis=1, inplace=True)
    return aux_features


def prepare_time_features(time_sequence_features, columns):
    cols = [col for col in time_sequence_features.columns if col not in columns]
    stripped_tsf = time_sequence_features[cols]
    tags_count = stripped_tsf.shape[1] - 1
    time = stripped_tsf.values.tolist()
    time = [make_structure_time_features(time[i]) for i in tqdm(range(len(time)))]
    time_pca = get_reduced_words_dimension(time, tags_count)

    print("Time features: ", stripped_tsf.columns)

    return time_pca, tags_count


def get_reduced_words_dimension(data, tags_count):
    """
    Get the aggregated dataset of words and tags which has the
    same dimensionality using PCA.
    :param: data: data which needs to be aggregated.
    """
    tags = np.array([i for i, _ in data])
    # Instantiating PCA to 35 components since it should be equal to the size of the vector of the tags
    # pca = PCA(n_components=35)
    pca = PCA(n_components=tags_count)
    word_embeddings = [j for _, j in data]
    pca.fit(word_embeddings)
    word_embeddings_pca = pca.transform(word_embeddings)
    tags = np.array(tags)
    return np.dstack((word_embeddings_pca, tags))


def prepare_citation_embedding(dataset, model):
    # Features for the LSTM - more time sequence related
    char_counts = pd.Series(Counter(chain.from_iterable(x for x in dataset.characters)))

    # Make a dictionary for creating a mapping between the char and the corresponding index
    char2ind = {char: i for i, char in enumerate(char_counts.index)}
    citation_layer = model.get_layer('citation_embedding')
    citation_weights = citation_layer.get_weights()[0]

    # Map the embedding of each character to the character in each corresponding citation and aggregate (sum)
    dataset['embedding'] = dataset['characters'].progress_apply(
        lambda x: sum([citation_weights[char2ind[c]] for c in x]))

    # Normalize the citation embeddings so that we can check for their similarity later
    dataset['embedding'] = dataset['embedding'].progress_apply(
        lambda x: x / np.linalg.norm(x, axis=0).reshape((-1, 1)))

    return dataset


# Generate word features
def prepare_citation_word_features(dataset, model):
    dataset['neighboring_words'] = dataset['neighboring_words'].progress_apply(
        lambda x: [i.lower() for i in x])

    # print("Checking words (row 200): ", dataset['neighboring_words'][200], type(dataset['neighboring_words'][200]))
    # print("Checking words (row 1000): ", dataset['neighboring_words'][1000], type(dataset['neighboring_words'][1000]))

    word_counts = pd.Series(Counter(chain.from_iterable(x for x in dataset.neighboring_words)))
    threshold = 4
    x = len(word_counts)
    y = len(word_counts[word_counts <= threshold])
    print('Total words: {}\nTotal number of words whose occurence is less than 4: {}\nDifference: {}'.format(x, y,
                                                                                                             x - y))
    words_less_than_threshold = word_counts[word_counts <= threshold]
    dataset['neighboring_words'] = dataset['neighboring_words'].progress_apply(
        lambda x: [i if i not in words_less_than_threshold else '<UNK>' for i in x]
    )
    dataset['neighboring_words'] = [['<UNK>'] if not x else x for x in
                                    dataset['neighboring_words']]

    words = pd.Series(Counter(chain.from_iterable(x for x in dataset.neighboring_words))).index
    word2ind = {w: i for i, w in enumerate(words)}
    word_embedding_matrix = np.zeros((len(word2ind), 300))

    for w in tqdm(word2ind):
        # word_embedding_matrix[word2ind[w]] = model.wv[w]
        word_embedding_matrix[word2ind[w]][:len(model.wv[w])] = model.wv[w]

    dataset['words_embedding'] = dataset['neighboring_words'].progress_apply(
        lambda x: sum([word_embedding_matrix[word2ind[w]] for w in x]))

    return dataset


# Select auxiliary features which are going to be used in the neural network
def encode_sections(dataset, largest_sections):
    # Change section to `OTHERS` if occurrence of the section is not in the largest sections
    dataset['sections'] = dataset['sections'].progress_apply(
        lambda x: list(set(['others' if i not in list(largest_sections) else i for i in x]))
    )
    section_dummies = pd.get_dummies(dataset['sections'].progress_apply(pd.Series).stack())
    residual_sections = set(list(largest_sections)) - set(section_dummies.columns)
    if len(residual_sections) > 0:
        for r_s in residual_sections:
            section_dummies[r_s] = 0
    dataset = dataset.join(section_dummies.sum(level=0))
    dataset.drop('sections', axis=1, inplace=True)
    print('Shape of auxiliary features after section generation: {}'.format(dataset.shape))
    print('Auxiliary features after section generation: {}'.format(dataset.columns))
    return dataset


def encode_citation_tag_features(dataset):
    # Taking the `neighboring_tags` and making an encoder dictionary for it
    # To have more info about how what tag mean what:
    # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

    # Get the count for each POS tag so that we have an estimation as to how many are there
    # tag_counts = pd.Series(Counter(chain.from_iterable(x for x in citation_tag_features.neighboring_tags)))
    # tag_counts.to_csv(TAG_COUNT, header=None)

    # We are going to replace `LS`, `the 2 backquotes` and the `the dollar symbol` since they do not have too much
    # use case and do not give too much information about the context of the neighboring citation text.
    OTHER_TAGS = ['LS', '``', "''", '$']

    dataset['neighboring_tags'] = dataset['neighboring_tags'].progress_apply(
        lambda x: [i if i not in OTHER_TAGS else 'others' for i in x]
    )
    # Now, we can use the `count vectorizer` to represent the `POS tags` as a vector where each element of the vector
    # represents the count of that tag in that particular citation.
    cv = CountVectorizer() # Instantiate the vectorizer
    dataset['neighboring_tags'] = dataset['neighboring_tags'].progress_apply(
        lambda x: " ".join(x))

    transformed_neighboring_tags = cv.fit_transform(dataset['neighboring_tags'])
    transformed_neighboring_tags = pd.DataFrame(transformed_neighboring_tags.toarray(), columns=cv.get_feature_names())

    print("Transformed neighboring tags dimensions: ", transformed_neighboring_tags.shape)
    print("Citation tag features dimensions: ", dataset.shape)

    dataset = dataset.reset_index(drop=True)
    dataset = pd.concat([dataset, transformed_neighboring_tags], join='inner', axis=1)
    dataset.drop('neighboring_tags', axis=1, inplace=True)

    return dataset

