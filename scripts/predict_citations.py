import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from collections import Counter
from keras.models import load_model
from gensim.models import FastText
from citation_classifier_helper import prepare_time_features, prepare_citation_word_features, \
    prepare_auxiliary_features, prepare_citation_tag_features, join_auxiliary_with_text


import warnings
warnings.filterwarnings("ignore")
tqdm.pandas()

np.random.seed(0)


def update_ids(x):
    kinds_of_ids = set()
    for item in x:
        kinds_of_ids.add(item[0])
    return kinds_of_ids


def prepare_time_sequence_features(tag_features, word_features):
    print("PREPARING TIME SEQUENCE FEATURES...")

    # Join time sequence features with the citations dataset
    time_sequence_features = pd.concat([tag_features, word_features
                                       .reset_index(drop=True)], keys=['id', 'citations'], axis=1)
    time_sequence_features = time_sequence_features.loc[:, ~time_sequence_features.columns.duplicated()]
    print('Total number of samples in time features are: {}'.format(time_sequence_features.shape))
    time_sequence_features = pd.concat([time_sequence_features['id'], time_sequence_features['citations']], axis=1)
    time_sequence_features['words_embedding'] = [np.array([]) if not isinstance(x, np.ndarray) else x for x in
                                                 time_sequence_features['words_embedding']]
    time_sequence_features['words_embedding'] = time_sequence_features['words_embedding'].progress_apply(
        lambda x: list(x))

    print("PREPARING TIME SEQUENCE FEATURES - DONE!")
    return time_sequence_features


def predict_citations(PROJECT_HOME, ext):
    print("Step FINAL : Predicting citations...")

    FASTTEXT_MODEL = '../data/model/{}wiki_fasttext.txt'.format(ext)
    CITATIONS_FEATURES = PROJECT_HOME + 'data/features/{}citations_features.parquet'.format(ext)

    NEWSPAPER_CITATIONS = PROJECT_HOME + 'data/features/{}newspaper_citation_features.parquet'.format(ext)
    LARGEST_SECTIONS = PROJECT_HOME + 'data/features/{}largest_sections.csv'.format(ext)
    RESULT_FILE = PROJECT_HOME + 'data/content/' + ext + 'result_{}.csv'

    TAG_COUNT = PROJECT_HOME + '/data/features/{}tag_counts.csv'.format(ext)
    CHAR_COUNT = PROJECT_HOME + '/data/features/{}char_counts.csv'.format(ext)

    MODEL_EMBEDDEDING = '../data/model/{}embedding_model.h5'.format(ext)
    MODEL_CITATION_EPOCHS_H5 = '../data/model/' + ext + 'citation_model_epochs_{}.h5'

    # Get the top 150 sections which we got from training the 2.7 million citations
    largest_sections = pd.read_csv(LARGEST_SECTIONS, header=None)
    largest_sections.rename({0: 'section_name', 1: 'count'}, axis=1, inplace=True)

    # original_tag_counts = pd.read_csv(TAG_COUNT, header=None)
    # original_tag_counts.rename({0: 'tag', 1: 'count'}, axis=1, inplace=True)

    # Load the pretrained embedding model on wikipedia
    model_fasttext = FastText.load(FASTTEXT_MODEL)

    model_embedding = load_model(MODEL_EMBEDDEDING)
    model = load_model(MODEL_CITATION_EPOCHS_H5.format(30))

    print('Loaded files and intermediary files...')

    newspaper_data = pd.read_parquet(NEWSPAPER_CITATIONS, engine='pyarrow')
    print('Loaded newspaper datasets...')

    def needs_a_label_or_not(row):
        """
            'ID_list' (0), 'citations' (1), 'type_of_citation'(2)
        """
        if row[1] in newspaper_data['citations']:
            return 'web'
        if not row[0]:
            return 'NO LABEL'

        # NK Book and journal labels can also be matched with entries in a preprocessed files
        id_list_str = list(
            item.split('=')
            for item in row[0].replace('{','').replace('}','').replace(' ', '').split(','))
        if len([i for i in ['PMC', 'PMID'] if i in update_ids(id_list_str)]) > 0:
            return 'journal'
        elif len([i for i in ['DOI'] if i in update_ids(id_list_str)]) == 1:
            if (len([i for i in ['DOI', 'ISBN'] if i in update_ids(id_list_str)]) == 2) and ('cite journal' in row[2]) and ('cite conference' in row[2]):
                return 'journal'
            elif (len([i for i in ['ISBN', 'DOI'] if i in update_ids(id_list_str)]) == 2) and ('cite book' in row[2]) and ('cite encyclopedia' in row[2]):
                return 'book'
            else:
                return 'journal'
        elif len([i for i in ['ISBN'] if i in update_ids(id_list_str)]) == 1:
            return 'book'
        else:
            return 'NO LABEL'

    # Generate text features
    def prepare_citation_text_features(auxiliary_features):
        citation_text_features = auxiliary_features['citations']

        # Convert the citation into a list by breaking it down into characters
        citation_text_features = citation_text_features.progress_apply(lambda x: list(x))

        char_counts = pd.Series(Counter(chain.from_iterable(x for x in citation_text_features)))

        # original_char_counts = pd.read_csv(CHAR_COUNT, header=None)

        char2ind = {char[0]: i for i, char in enumerate(char_counts.index)}

        citation_layer = model_embedding.get_layer('citation_embedding')
        citation_weights = citation_layer.get_weights()[0]
        citation_text_features = citation_text_features.to_frame()

        # Map the embedding of each character to the character in each corresponding citation and aggregate (sum)
        citation_text_features['embedding'] = citation_text_features['citations'].progress_apply(
            lambda x: sum([(citation_weights[char2ind[c]] if c in char2ind else 0) for c in x]))

        # Normalize the citation embeddings so that we can check for their similarity later
        citation_text_features['embedding'] = citation_text_features['embedding'].progress_apply(
            lambda x: x / np.linalg.norm(x, axis=0).reshape((-1, 1)))
        return citation_text_features

    FILES = os.listdir(CITATIONS_FEATURES)

    for index__, f_name in enumerate(FILES):
        if f_name == '_SUCCESS':
            continue
        if f_name.endswith('.crc'):
            continue
        f_name_path = '{}/{}'.format(CITATIONS_FEATURES, f_name)
        all_examples = pd.read_parquet(f_name_path, engine='pyarrow')

        # TODO NK Remove
        all_examples = all_examples.head(400000)

        print('Doing filename: {} with citations: {}'.format(f_name, all_examples.shape[0]))
        all_examples['real_citation_text'] = all_examples['citations']
        all_examples['needs_a_label'] = all_examples[['ID_list', 'citations', 'type_of_citation']].progress_apply(
            lambda x: needs_a_label_or_not(x), axis=1)
        not_wild_examples = all_examples[all_examples['needs_a_label'] != 'NO LABEL'].reset_index(drop=True)
        wild_examples = all_examples[all_examples['needs_a_label'] == 'NO LABEL'].reset_index(drop=True)
        print('Preprocessing the citations for wild examples')
        print(all_examples.shape, wild_examples.shape, not_wild_examples.shape)

        # Remove the biases
        labels = ['doi', 'isbn', 'pmc', 'pmid', 'url', 'work', 'newspaper', 'website']

        for label in labels:
            wild_examples['citations'] = wild_examples['citations'].apply(
                lambda x: re.sub(label + '\s{0,10}=\s{0,10}([^|]+)', label + ' = ', x))
        print('Number of wild citations in this file: {}'.format(wild_examples.shape))

        print('Any sections in the parent section: {}'.format(
            not any([True if i in list(largest_sections['section_name']) else False for i in set(wild_examples['sections'])])))

        auxiliary_features = wild_examples[
            ['sections', 'citations', 'ref_index', 'neighboring_words', 'total_words', 'neighboring_tags']]
        auxiliary_features = prepare_auxiliary_features(auxiliary_features, largest_sections['section_name'])

        citation_tag_features = prepare_citation_tag_features(auxiliary_features[['citations', 'neighboring_tags']])

        citation_text_features = prepare_citation_text_features(auxiliary_features)

        citation_word_features = auxiliary_features[['citations', 'neighboring_words']]
        citation_word_features = prepare_citation_word_features(citation_word_features, model_fasttext)

        # Join auxiliary features with the citations dataset
        auxiliary_features = join_auxiliary_with_text(auxiliary_features, citation_text_features)
        auxiliary_features.drop(['neighboring_tags'], axis=1, inplace=True)

        time_sequence_features = prepare_time_sequence_features(citation_tag_features, citation_word_features)

        time_pca, tags_count = prepare_time_features(time_sequence_features, ['citations', 'neighboring_words'])

        # Classify citations

        # Make a mask for auxiliary dataset to get all features except the one below
        column_mask_aux = ~auxiliary_features.columns.isin(['citations', 'neighboring_words'])
        auxiliary = auxiliary_features.loc[:, column_mask_aux].values.tolist()
        auxiliary = [np.array(auxiliary[i][0][0] + auxiliary[i][1:]) for i in tqdm(range(len(auxiliary)))]

        print('Features for model constructed.. now running model')

        # RUN MODEL
        # print("Input data (time): ", len(time_pca[100]))
        # print("Input data (aux): ", len(np.array(auxiliary)[100]))

        prediction = model.predict([time_pca, np.array(auxiliary)])
        print('Shape of prediction: {}'.format(prediction.shape))
        y_pred = np.argmax(prediction, axis=1)
        wild_examples['label_category'] = y_pred
        print('Done with model prediction for index: {}'.format(index__))

        # Result saved
        not_wild_examples['label_category'] = None
        columns = ['id', 'page_title', 'real_citation_text', 'ID_list', 'type_of_citation', 'label_category', 'needs_a_label']
        resultant_examples = pd.concat([wild_examples[columns], not_wild_examples[columns]]).reset_index(drop=True)
        resultant_examples.rename({'label_category': 'predicted_label_no', 'needs_a_label': 'existing_label', 'real_citation_text': 'citations'}, axis=1, inplace=True)
        print('Saving a file with f_name: {} with citations: {} with all:{} and wild: {} and non-wild: {}'.format(
            f_name, resultant_examples.shape[0], all_examples.shape[0], wild_examples.shape[0], not_wild_examples.shape[0]))

        resultant_examples.to_csv(RESULT_FILE.format(index__), index=False, encoding='utf-8')
        print('\nFile saved for part: {}\n\n'.format(index__))


# PROJECT_HOME = "gs://wikicite-1/"
PROJECT_HOME = 'c:///users/natal/PycharmProjects/cite-classifications-wiki/'
ext = "en_"

predict_citations(PROJECT_HOME, ext)