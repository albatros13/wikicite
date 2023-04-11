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
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from keras.utils import to_categorical
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from citation_classifier_helper import prepare_citation_word_features, encode_sections, \
    join_auxiliary_with_text, encode_citation_tag_features, prepare_time_features, \
    prepare_citation_embedding, encode_auxuliary, encode_citation_type, embed_citation_text


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

    # NK take sample for huge datasets
    # newspaper_data = newspaper_data.sample(n=1000000)

    newspaper_data.drop('type_of_citation', axis=1, inplace=True)
    book_journal_features.drop('type_of_citation', axis=1, inplace=True)

    book_features.drop('type_of_citation', axis=1, inplace=True)
    journal_features.drop('type_of_citation', axis=1, inplace=True)

    dataset_with_features = pd.concat([journal_features, book_features, newspaper_data])

    le = preprocessing.LabelEncoder()
    le.fit(dataset_with_features['actual_label'])
    dataset_with_features['label_category'] = le.transform(dataset_with_features['actual_label'])

    print("BOOK:", dataset_with_features[dataset_with_features['actual_label'] == 'book'].head(1))
    print("WEB:", dataset_with_features[dataset_with_features['actual_label'] == 'web'].head(1))
    print("JOURNAL:", dataset_with_features[dataset_with_features['actual_label'] == 'journal'].head(1))

    # Clearing up memory
    del book_journal_features
    del newspaper_data
    # Remove rows which have duplicate ID and citations since they are just the same examples
    dataset_with_features = dataset_with_features.drop_duplicates(subset=['id', 'citations'])
    dataset_with_features = dataset_with_features.reset_index(drop=True)
    print("Final labelled dataset dimensions: ", dataset_with_features.shape)
    return dataset_with_features


def prepare_time_sequence_features(tag_features, word_features, data):
    # Join time sequence features with the citations dataset
    time_seq_features = pd.concat([tag_features, word_features.reset_index(drop=True)],
                                       keys=['id', 'citations'], axis=1)
    time_seq_features = time_seq_features.loc[:, ~time_seq_features.columns.duplicated()]
    print('Total number of samples in time features are: {}'.format(time_seq_features.shape))

    time_seq_features = pd.concat([time_seq_features['id'], time_seq_features['citations']], axis=1)

    time_seq_features = pd.concat([time_seq_features, data.reset_index(drop=True)],
                                       keys=['id', 'citations'], axis=1)
    time_seq_features.columns = time_seq_features.columns.droplevel(0)

    time_seq_features = time_seq_features.loc[:, ~time_seq_features.columns.duplicated()]
    time_seq_features['words_embedding'] = time_seq_features['words_embedding'].progress_apply(
        lambda x: [x] if isinstance(x, int) else x.tolist())

    return time_seq_features


def citation_embedding_model(input_dim):
    """
    Citation embedding generator model where the dimension of the embedding is 50.
    """
    main_input = Input(shape=(400,), name='characters')
    # input_dim is basically the vocab size
    emb = Embedding(input_dim=input_dim, output_dim=300, name='citation_embedding')(main_input)
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


def tsne_embedding_plot(text_features):
    citation_text_and_embeddings = text_features[['citations', 'embedding']][:500]
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
    model = load_model(EMBEDDING_MODEL)

    # model = citation_embedding_model(312)
    #
    # # NK The argument steps_per_epoch should be equal to the total number of samples (length of your training set)
    # # divided by batch_size
    # BATCH_SIZE = 8
    # steps = len(training_data) #
    # print("Steps per epoch: ", steps)
    # # Run the model with the data being generated by the generator with a batch size of 64
    # # and number of epochs to be set to 15
    # # hist = model.fit_generator(generator(training_data, categorical_labels, 512), steps_per_epoch=4000, epochs=2)
    # model.fit_generator(generator(training_data, categorical_labels, BATCH_SIZE), steps_per_epoch=steps, epochs=2)

    # Evaluation of embedding model
    print("Testing data shape:", np.array(testing_data).shape)

    y_predicted_proba = model.predict(np.array(testing_data))
    predicted_class = np.argmax(y_predicted_proba, axis=1)
    accuracy = accuracy_score(testing_labels, predicted_class)
    print("Accuracy of the embedding model: {}".format(accuracy))

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


def train_classification_model(tags_count, auxiliary, time_pca, label_categories):
    # We use `ReduceLRonPlateau` so that the model does not overshoot the optimal minimum point and hence by default we
    # start with a learning rate of 0.01 but as soon as the accuracy stop increasing the learning rate does not change
    # which helps us converge better.

    # Get the labels which will be split later
    y = np.asarray(label_categories)

    # EPOCHS = 30
    # NK for faster testing
    EPOCHS = 3

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

    BATCH_SIZE = 256
    # BATCH_SIZE = 8
    print('Running model with epochs: {}'.format(EPOCHS))

    # Reuse model for quick testing
    # model = load_model(MODEL_CITATION_EPOCHS_H5.format(EPOCHS))

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


def print_stats(aux_features, text_features):
    print('Total mean length of book articles: {}'.format( # Books - length is less
        aux_features[aux_features['label_category'] == 0]['total_words'].mean()))
    print('Total median length of book articles: {}'.format(
        aux_features[aux_features['label_category'] == 0]['total_words'].median()))

    print('Total mean length of journal articles: {}'.format(
        aux_features[aux_features['label_category'] == 1]['total_words'].mean()))
    print('Total median length of journal articles: {}'.format(
        aux_features[aux_features['label_category'] == 1]['total_words'].median()))

    print('Total mean length of news articles: {}'.format(
        aux_features[aux_features['label_category'] == 2]['total_words'].mean()))
    print('Total median length of news articles: {}'.format(
        aux_features[aux_features['label_category'] == 2]['total_words'].median()))

    # Get the character counts for each unique character
    print('The max length of the longest citation in terms of characters is: {}'.format(
        max(text_features.characters.apply(lambda x: len(x)))))
    print('The mean length of the longest citation in terms of characters is: {}'.format(
        text_features.characters.apply(lambda x: len(x)).mean()))
    print('The median length of the longest citation in terms of characters is: {}'.format(
        text_features.characters.apply(lambda x: len(x)).median()))


# MAIN
if __name__ == '__main__':

    # dataset_with_features = prepare_labelled_dataset()
    # dataset_with_features.to_csv(DATASET_WITH_FEATURES, index=False)

    # re-load labelled dataset
    dataset_with_features = pd.read_csv(DATASET_WITH_FEATURES)

    # NK Convert strings to arrays
    for col in ['neighboring_tags', 'neighboring_words', 'sections']:
        dataset_with_features[col] = dataset_with_features[col].progress_apply(
             lambda x: x.replace("'", "").replace('[', '').replace(']', '').replace('\n', '').split(','))

    # 'sections', 'citations', 'id', 'ref_index', 'total_words', 'neighboring_tags', 'label_category'
    auxiliary_features = dataset_with_features[
            ['sections', 'citations', 'id', 'ref_index', 'total_words', 'neighboring_tags', 'label_category']]

    # section_counts = pd.Series(Counter(chain.from_iterable(x for x in auxiliary_features.sections)))
    # largest_sections = section_counts.nlargest(150)
    # largest_sections.to_csv(LARGEST_SECTIONS, header=None)

    # re-load largest sections
    largest_sections = pd.read_csv(LARGEST_SECTIONS, header=None)
    largest_sections.rename({0: 'section_name', 1: 'count'}, axis=1, inplace=True)

    # largest_sections.index
    auxiliary_features = encode_sections(auxiliary_features, largest_sections['section_name'])
    auxiliary_features = encode_citation_type(auxiliary_features)

    citation_tag_features = encode_citation_tag_features(
        auxiliary_features[['id', 'citations', 'neighboring_tags']])

    citation_text_features = auxiliary_features[['id', 'citations', 'label_category']]
    # Convert the citation into a list by breaking it down into characters
    citation_text_features['characters'] = citation_text_features['citations'].progress_apply(lambda x: list(x))

    print_stats(auxiliary_features, citation_text_features)

    data = embed_citation_text(citation_text_features)

    model = train_model(data)
    citation_text_features = prepare_citation_embedding(citation_text_features, model)

    # This is optional, just to see a plot
    # tsne_embedding_plot(citation_text_features)

    model = FastText.load(FASTTEXT_MODEL)
    citation_word_features = prepare_citation_word_features(
        dataset_with_features[['id', 'citations', 'neighboring_words', 'label_category']], model)

    # Join auxiliary features with the citations dataset and encode
    auxiliary_features = join_auxiliary_with_text(auxiliary_features, citation_text_features)
    auxiliary = encode_auxuliary(auxiliary_features)

    time_sequence_features = prepare_time_sequence_features(
        citation_tag_features, citation_word_features, dataset_with_features[['id', 'citations', 'label_category']]
    )

    time_pca, tags_count = prepare_time_features(time_sequence_features, ['id', 'citations', 'label_category', 'neighboring_words'])

    label_categories = auxiliary_features.loc[:, 'label_category'].astype(int).tolist()
    train_classification_model(tags_count, auxiliary, time_pca, label_categories)
