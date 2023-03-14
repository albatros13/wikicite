import re
import gc
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from collections import Counter
from gensim.models import FastText
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import keras
from keras.models import Model, load_model
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from keras.utils import to_categorical
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from citation_classifier_helper import prepare_citation_word_features, prepare_auxiliary_features, \
    join_auxiliary_with_text, prepare_citation_tag_features, prepare_time_features


ext = "en_"
PROJECT_HOME = 'c:///users/natal/PycharmProjects/cite-classifications-wiki/'

BOOK_JOURNAL_CITATIONS = PROJECT_HOME + 'data/features/{}book_journal_citations.parquet'.format(ext)
NEWSPAPER_CITATIONS = PROJECT_HOME + 'data/features/{}newspaper_citation_features.parquet'.format(ext)

TAG_COUNT = PROJECT_HOME + 'data/features/{}tag_counts.csv'.format(ext)
CHAR_COUNT = PROJECT_HOME + 'data/features/{}char_counts.csv'.format(ext)
LARGEST_SECTIONS = PROJECT_HOME + 'data/features/{}largest_sections.csv'.format(ext)

DATASET_WITH_FEATURES = PROJECT_HOME + 'data/features/{}dataset_with_features.csv'.format(ext)

FASTTEXT_MODEL = '../data/model/{}wiki_fasttext.txt'.format(ext)
EMBEDDING_MODEL = '../data/model/{}embedding_model.h5'.format(ext)
MODEL_CITATION_LOSS = '../data/model/'+ext+'citation_model_loss_{}.json'
MODEL_CITATION_RESULT = '../data/model/'+ext+'citation_model_result_{}.json'
MODEL_CITATION_EPOCHS_H5 = '../data/model/'+ext+'citation_model_epochs_{}.h5'
MODEL_CITATION_EPOCHS_JSON = '../data/model/'+ext+'citation_model_epochs_{}.json'

gc.collect()
warnings.filterwarnings("ignore")
tqdm.pandas()
keras.backend.backend()

np.random.seed(0)

# Get auxiliary features and divide them into labels
# 1. `ref_index`
# 2. `total_words`
# 3. `tags`
# 4. `type_of_citation`
# Can include `section` of the page in which the citation belongs to


def prepare_labelled_dataset():
    print("PREPARING LABELLED DATASET...")

    # BOOKS AND JOURNALS
    book_journal_features = pd.read_parquet(BOOK_JOURNAL_CITATIONS, engine='pyarrow')

    labels = ['doi', 'isbn', 'pmc', 'pmid', 'url', 'work', 'newspaper', 'website']
    for label in labels:
        book_journal_features['citations'] = book_journal_features['citations'].progress_apply(
            lambda x: re.sub(label + '\s{0,10}=\s{0,10}([^|]+)', label + ' = ', x))

    book_journal_features['actual_label'].value_counts()

    journal_features = book_journal_features[book_journal_features['actual_label'] == 'journal']
    book_features = book_journal_features[book_journal_features['actual_label'] == 'book']

    print('The total number of journals: {}'.format(journal_features.shape))
    print('The total number of books: {}'.format(book_features.shape))

    # NEWSPAPERS

    newspaper_data = pd.read_parquet(NEWSPAPER_CITATIONS, engine='pyarrow')
    print('The total number of newspapers: {}'.format(newspaper_data.shape))

    newspaper_data = newspaper_data[[
        'citations', 'ref_index', 'total_words', 'neighboring_words', 'neighboring_tags', 'id', 'sections', 'type_of_citation'
    ]]
    newspaper_data['actual_label'] = 'web'
    for label in labels:
        newspaper_data['citations'] = newspaper_data['citations'].progress_apply(
            lambda x: re.sub(label + '\s{0,10}=\s{0,10}([^|]+)', label + ' = ', x))

    # NK TODO take sample for huge datasets?
    # newspaper_data = newspaper_data.sample(n=n)

    newspaper_data.drop('type_of_citation', axis=1, inplace=True)
    book_journal_features.drop('type_of_citation', axis=1, inplace=True)

    book_features.drop('type_of_citation', axis=1, inplace=True)
    journal_features.drop('type_of_citation', axis=1, inplace=True)

    dataset_with_features = pd.concat([journal_features, book_features, newspaper_data])

    le = preprocessing.LabelEncoder()
    le.fit(dataset_with_features['actual_label'])
    dataset_with_features['label_category'] = le.transform(dataset_with_features['actual_label'])
    # dataset_with_features[dataset_with_features['actual_label'] == 'entertainment'].head(1)
    dataset_with_features[dataset_with_features['actual_label'] == 'web'].head(1)
    dataset_with_features[dataset_with_features['actual_label'] == 'book'].head(1)
    dataset_with_features[dataset_with_features['actual_label'] == 'journal'].head(1)
    # Clearing up memory
    del book_journal_features
    del newspaper_data
    # Remove rows which have duplicate ID and citations since they are just the same examples
    dataset_with_features = dataset_with_features.drop_duplicates(subset=['id', 'citations'])
    dataset_with_features = dataset_with_features.reset_index(drop=True)
    print("Final labelled dataset dimensions: ", dataset_with_features.shape)
    print("PREPARING LABELLED DATASET - DONE!")
    return dataset_with_features


def encode_citation_type(auxiliary_features):
    # Get one hot encoding of citation_type column
    if 'citation_type' in auxiliary_features:
        citation_type_encoding = pd.get_dummies(auxiliary_features['citation_type'])
        # Drop column citation_type as it is now encoded and join it
        auxiliary_features = auxiliary_features.drop('citation_type', axis=1)
        # Concat columns of the dummies along the axis with the matching index
        auxiliary_features = pd.concat([auxiliary_features, citation_type_encoding], axis=1)

    print('Total mean length of entertainment articles: {}'.format(
        auxiliary_features[auxiliary_features['label_category'] == 1]['total_words'].mean()))
    print('Total median length of entertainment articles: {}'.format(
        auxiliary_features[auxiliary_features['label_category'] == 1]['total_words'].median()))

    print('Total mean length of journal articles: {}'.format(
        auxiliary_features[auxiliary_features['label_category'] == 2]['total_words'].mean()))
    print('Total median length of journal articles: {}'.format(
        auxiliary_features[auxiliary_features['label_category'] == 2]['total_words'].median()))

    print('Total mean length of book articles: {}'.format( # Books - length is less
        auxiliary_features[auxiliary_features['label_category'] == 0]['total_words'].mean()))
    print('Total median length of book articles: {}'.format(
        auxiliary_features[auxiliary_features['label_category'] == 0]['total_words'].median()))

    return auxiliary_features


def prepare_citation_text_features(dataset_with_features):
    print("PREPARING CITATION_TEXT_FEATURES...")
    # Features for the LSTM - more time sequence related
    # Citation's original text features

    # Create a separate dataframe for preprocessing citation text
    citation_text_features = dataset_with_features[['id', 'citations', 'label_category']]

    # Convert the citation into a list by breaking it down into characters
    citation_text_features['characters'] = citation_text_features['citations'].progress_apply(lambda x: list(x))

    # Get the character counts for each unique character
    print('The max length of the longest citation in terms of characters is: {}'.format(
        max(citation_text_features.characters.apply(lambda x: len(x)))))
    print('The mean length of the longest citation in terms of characters is: {}'.format(
        citation_text_features.characters.apply(lambda x: len(x)).mean()))
    print('The median length of the longest citation in terms of characters is: {}'.format(
        citation_text_features.characters.apply(lambda x: len(x)).median()))

    print("PREPARING CITATION_TEXT_FEATURES - DONE")
    return citation_text_features


def prepare_data(citation_text_features):
    print("PREPARING DATA...")
    char_counts = pd.Series(Counter(chain.from_iterable(x for x in citation_text_features.characters)))
    # char_counts.to_csv(CHAR_COUNT, header=None)
    # Make a dictionary for creating a mapping between the char and the corresponding index
    char2ind = {char: i for i, char in enumerate(char_counts.index)}
    # ind2char = {i: char for i, char in enumerate(char_counts.index)}

    # Map each character into the citation to its corresponding index and store it in a list
    X_char = []
    for citation in citation_text_features.citations:
        citation_chars = []
        for character in citation:
            citation_chars.append(char2ind[character])
        X_char.append(citation_chars)

    # Since the median length of the citation is 282, we have padded the input till 400 to get extra information which
    # would be fed into the character embedding neural network.
    X_char = pad_sequences(X_char, maxlen=400)

    # Append the citation character list with their corresponding lists for making a dataset
    # for getting the character embeddings
    data = []
    for i in tqdm(range(len(X_char))):
        data.append((X_char[i], int(citation_text_features.iloc[i]['label_category'])))
    print("Final data dimensions: ", len(data), len(data[0]))
    print("PREPARING DATA - DONE!")
    return data


def prepare_time_sequence_features(tag_features, word_features, data):
    print("PREPARING TIME SEQUENCE FETURES...")

    # Join time sequence features with the citations dataset
    time_sequence_features = pd.concat([tag_features, word_features.reset_index(drop=True)],
                                       keys=['id', 'citations'], axis=1)
    time_sequence_features = time_sequence_features.loc[:, ~time_sequence_features.columns.duplicated()]
    print('Total number of samples in time features are: {}'.format(time_sequence_features.shape))

    # Join the time sequence features for the data
    time_sequence_features = pd.concat([time_sequence_features['id'], time_sequence_features['citations']], axis=1)

    time_sequence_features = pd.concat([time_sequence_features, data.reset_index(drop=True)],
                                       keys=['id', 'citations'], axis=1)
    time_sequence_features.columns = time_sequence_features.columns.droplevel(0)

    time_sequence_features = time_sequence_features.loc[:, ~time_sequence_features.columns.duplicated()]
    time_sequence_features['words_embedding'] = time_sequence_features['words_embedding'].progress_apply(
        lambda x: [x] if isinstance(x, int) else x.tolist())

    print("PREPARING TIME SEQUENCE FEATURES - DONE!")
    return time_sequence_features


def citation_embedding_model():
    """
    Citation embedding generator model where the dimension of the embedding is 50.
    """
    main_input = Input(shape=(400,), name='characters')
    # input dim is basically the vocab size
    # emb = Embedding(input_dim=95, output_dim=300, name='citation_embedding')(main_input)
    emb = Embedding(input_dim=312, output_dim=300, name='citation_embedding')(main_input)
    rnn = Bidirectional(LSTM(20))
    x = rnn(emb)
    de = Dense(3, activation='softmax')(x)
    model = Model(inputs=main_input, outputs=de)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def generator(features, categorical_labels, batch_size):
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


def prepare_citation_embedding(citation_text_features, model):
    print("PREPARING CITATION EMBEDDING...")
    char_counts = pd.Series(Counter(chain.from_iterable(x for x in citation_text_features.characters)))

    # Make a dictionary for creating a mapping between the char and the corresponding index
    char2ind = {char: i for i, char in enumerate(char_counts.index)}
    citation_layer = model.get_layer('citation_embedding')
    citation_weights = citation_layer.get_weights()[0]
    # print("char2ind: ", char2ind)

    # Map the embedding of each character to the character in each corresponding citation and aggregate (sum)
    citation_text_features['embedding'] = citation_text_features['characters'].progress_apply(
        lambda x: sum([citation_weights[char2ind[c]] for c in x]))

    # Normalize the citation embeddings so that we can check for their similarity later
    citation_text_features['embedding'] = citation_text_features['embedding'].progress_apply(
        lambda x: x / np.linalg.norm(x, axis=0).reshape((-1, 1)))

    # Make the sum of the embedding to be summed up to 1
    np.sum(np.square(citation_text_features['embedding'].iloc[0]))

    # Just considering 20 since otherwise it will be computationally extensive
    # citation_text_and_embeddings = citation_text_features[['citations', 'embedding']][:500]
    # citation_text_and_embeddings['embedding'] = citation_text_and_embeddings['embedding'].progress_apply(
    #     lambda x: x[0].tolist()
    # )
    print("PREPARING CITATION EMBEDDING - DONE!")
    return citation_text_features


def tsne_embedding_plot(citation_text_and_embeddings):
    labels = []
    tokens = []

    index = 0
    for row in citation_text_and_embeddings:
        tokens.append(row['embedding'])
        labels.append(str(index))
        index += 1

    # Perplexity takes into account the global and local features
    # We are using dimensionality reduction for 2 features and taking 2500 iterations into account
    tsne_model = TSNE(perplexity=40, n_components=2, n_iter=2500, random_state=0)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(10, 10))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')
    plt.show()


# LSTM/Neural Network Model
def generator_nn(features_aux, features_time, labels, batch_size):
    """
    Generator to create batches of data so that processing is easy.

    :param: features: the features of the model.
    :param: labels: the labels of the model.
    :param: batch_size: the size of the batch
    """
    # Create empty arrays to contain batch of features and labels
    # batch_features_aux = np.zeros((batch_size, 453))
    # batch_features_time =  np.zeros((batch_size, 35, 2))
    print("generator_nn dimensions:", features_aux.shape, features_time.shape)

    batch_features_aux = np.zeros((batch_size, features_aux.shape[1]))
    batch_features_time = np.zeros((batch_size, features_time.shape[1], 2))
    batch_labels = np.zeros((batch_size, 3))

    while True:
        for i in range(batch_size):
            # choose random index in features
            index = np.random.choice(len(features_aux), 1)[0]
            batch_features_aux[i] = features_aux[index]
            batch_features_time[i] = features_time[index]
            batch_labels[i] = labels[index]
        yield [batch_features_time, np.asarray(batch_features_aux)], batch_labels


def scheduler(epoch, lr):
    import math
    if epoch < 10:
        return lr
    else:
        return lr * math.exp(-0.1)


def train_model(data):
    # Separate out the training data and labels for further verification use
    features = [i[0] for i in data]
    labels = [i[1] for i in data]
    # Changing it to dummy labels - identifier vs non identifier
    labels = [i for i in labels]

    Counter(labels)

    # Splitting the data into training and testing
    training_data, testing_data, training_labels, testing_labels = train_test_split(
        features, labels, train_size=0.9, shuffle=True
    )

    # We are going to feed in the 400 character input since our median length comes out to be approximately 282 and train it
    # on a dummy task - if the citation is scientific or not and get the embedding layer which would contain the
    # representation for each character.

    categorical_labels = to_categorical(training_labels, num_classes=3)
    # categorical_test_labels = to_categorical(testing_labels, num_classes=3)

    # Instantiate the model and generate the summary
    model = citation_embedding_model()

    # NK The argument steps_per_epoch should be equal to the total number of samples (length of your training set)
    # divided by batch_size
    BATCH_SIZE = 8
    steps = len(training_data) # // batch_size
    print("Steps per epoch: ", steps)
    # Run the model with the data being generated by the generator with a batch size of 64
    # and number of epochs to be set to 15
    # hist = model.fit_generator(generator(training_data, categorical_labels, 512), steps_per_epoch=4000, epochs=2)
    model.fit_generator(generator(training_data, categorical_labels, BATCH_SIZE), steps_per_epoch=steps, epochs=2)

    # Evaluation of embedding model
    print(np.array(testing_data).shape)

    y_predicted_proba = model.predict(np.array(testing_data))
    predicted_class = np.argmax(y_predicted_proba, axis=1)
    accuracy_score(testing_labels, predicted_class)

    # Save the model so that we can retrieve it later
    model.save(EMBEDDING_MODEL)
    return model


def classification_model(tags_count, input_length):
    """
    Model for classifying whether a citation is scientific or not.
    """
    # main_input = Input(shape=(35, 2), name='time_input')
    main_input = Input(shape=(tags_count, 2), name='time_input')

    lstm_out = LSTM(64)(main_input)

    # auxiliary_input = Input(shape=(453,), name='aux_input') ## 454 without citation type, 476 with citation type
    auxiliary_input = Input(shape=(input_length,), name='aux_input')

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
        loss_weights={'main_output': 1.}, metrics=['acc'],
        run_eagerly=True
    )
    return model


def train_classification_model(tags_count, auxiliary_features, time_pca):
    # We use `ReduceLRonPlateau` so that the model does not overshoot the optimal minimum point and hence by default we
    # start with a learning rate of 0.01 but as soon as the accuracy stop increasing the learning rate does not change
    # which helps us converge better.

    # Make a mask for auxiliary dataset to get all features except the one below
    column_mask_aux = ~auxiliary_features.columns.isin(['id', 'citations', 'label_category'])
    # Get the columns of those auxiliary features and covert them into a list
    auxiliary = auxiliary_features.loc[:, column_mask_aux].values.tolist()
    # Convert them into numpy array (for Keras) and stack them (if needed) as suited for the model's format
    auxiliary = [np.array(auxiliary[i][0][0] + auxiliary[i][1:]) for i in tqdm(range(len(auxiliary)))]
    auxiliary = np.asarray(auxiliary)

    # Get the labels which will be split later
    y = auxiliary_features.loc[:, 'label_category'].astype(int).tolist()
    y = np.asarray(y)

    EPOCHS = 30

    x_train_indices, x_test_indices, y_train_indices, y_test_indices = train_test_split(
        range(auxiliary.shape[0]), range(y.shape[0]), train_size=0.9, stratify=y, shuffle=True
    )

    aux_train = auxiliary[x_train_indices]
    time_train = time_pca[x_train_indices]
    y_train = np.eye(3)[y[x_train_indices]]

    aux_test = auxiliary[x_test_indices]
    time_test = time_pca[x_test_indices]
    y_test = y[x_test_indices]

    print("time_train", time_train.shape)

    # BATCH_SIZE = 256
    BATCH_SIZE = 8
    print('Running model with epochs: {}'.format(EPOCHS))

    # NK reuse model for quick testing
    # model = load_model(MODEL_CITATION_EPOCHS_H5.format(30))

    print("Dimensions: ", aux_train.shape[1])

    model = classification_model(tags_count, aux_train.shape[1])
    training_generator = generator_nn(aux_train, time_train, y_train, BATCH_SIZE)
    steps = len(x_train_indices)

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history_callback = model.fit_generator(
        training_generator, steps_per_epoch=steps, epochs=EPOCHS, verbose=1, shuffle=True, callbacks=[callback]
    )

    history_dict = history_callback.history

    f = open(MODEL_CITATION_LOSS.format(EPOCHS), 'w')
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

    print('\n\nDone with the prediction and saving model with epochs: {}\n'.format(EPOCHS))

    res.to_csv(MODEL_CITATION_RESULT.format(EPOCHS))
    json_string = model.to_json()
    with open(MODEL_CITATION_EPOCHS_JSON.format(EPOCHS), "w") as json_file:
        json_file.write(json_string)
    model.save(MODEL_CITATION_EPOCHS_H5.format(EPOCHS))


# MAIN
if __name__ == '__main__':

    # dataset_with_features = prepare_labelled_dataset()
    # dataset_with_features.to_csv(DATASET_WITH_FEATURES, index=False)

    # re-load labelled dataset
    dataset_with_features = pd.read_csv(DATASET_WITH_FEATURES)

    # Convert strings to arrays
    for col in ['neighboring_tags', 'neighboring_words']:
        dataset_with_features[col] = dataset_with_features[col].progress_apply(
             lambda x: x.replace("'", "").replace('[', '').replace(']', '').replace('\n', '').split(','))

    auxiliary_features = dataset_with_features[
            ['sections', 'citations', 'id', 'ref_index', 'total_words', 'neighboring_tags', 'label_category']]
    # section_counts = pd.Series(Counter(chain.from_iterable(x for x in auxiliary_features.sections)))
    # largest_sections = section_counts.nlargest(150)
    # largest_sections.to_csv(LARGEST_SECTIONS, header=None)

    # re-load largest sections
    largest_sections = pd.read_csv(LARGEST_SECTIONS, header=None)

    auxiliary_features = prepare_auxiliary_features(auxiliary_features, largest_sections)
    auxiliary_features = encode_citation_type(auxiliary_features)

    citation_tag_features = prepare_citation_tag_features(
        auxiliary_features[['id', 'citations', 'neighboring_tags']])

    citation_text_features = prepare_citation_text_features(dataset_with_features)

    # data = prepare_data(citation_text_features)
    # model = train_model(data)
    model = load_model(EMBEDDING_MODEL)

    citation_text_features = prepare_citation_embedding(citation_text_features, model)

    # This is optional, just to see a plot
    # citation_text_and_embeddings = citation_text_features[['citations', 'embedding']][:500]
    # tsne_embedding_plot(citation_text_and_embeddings)

    model = FastText.load(FASTTEXT_MODEL)
    citation_word_features = dataset_with_features[['id', 'citations', 'neighboring_words', 'label_category']]
    citation_word_features = prepare_citation_word_features(citation_word_features, model)

    auxiliary_features = join_auxiliary_with_text(auxiliary_features, citation_text_features)
    auxiliary_features.drop(['neighboring_tags', 'characters'], axis=1, inplace=True)

    time_sequence_features = prepare_time_sequence_features(
        citation_tag_features, citation_word_features, dataset_with_features[['id', 'citations', 'label_category']]
    )

    time_pca, tags_count = prepare_time_features(time_sequence_features, ['id', 'citations', 'label_category', 'neighboring_words'])

    train_classification_model(tags_count, auxiliary_features, time_pca)
