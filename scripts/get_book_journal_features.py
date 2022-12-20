import re
import pandas as pd


def get_book_journal_features(file_in, file_out, file_out_csv):
    print("Step 7: Getting book and journal citations...")

    citations_features = pd.read_parquet(file_in, engine='pyarrow')

    print(type(citations_features))
    print("Before filter:", len(citations_features))
    print('The columns in the citations features are: {}'.format(citations_features.columns))
    citation_with_ids = citations_features[citations_features['ID_list'].notnull()]
    print("After filter:", len(citations_features))

    print('Total citations with NOT-NULL ID LIST: {}'.format(len(citation_with_ids)))

    # Formulate a structure for the ID_List in which we can do something meaningful

    # citation_with_ids['ID_list'] = citation_with_ids['ID_list'].progress_apply(
    #     lambda x: list(item.split('=') for item in x.replace('{','').replace('}','').replace(' ', '').split(',')))
    citation_with_ids['ID_list'] = citation_with_ids['ID_list'].apply(
        lambda x: list(item.split('=') for item in x.replace('{','').replace('}','').replace(' ', '').split(',')))

    # Get the kinds of ids associated with each tuple
    kinds_of_ids = set()

    def update_ids(x):
        for item in x:
            kinds_of_ids.add(item[0])

    # _ = citation_with_ids['ID_list'].progress_apply(lambda x: update_ids(x))
    _ = citation_with_ids['ID_list'].apply(lambda x: update_ids(x))

    print("KINDS_OF_IDS", kinds_of_ids)

    # Add the columns with NoneType in the previous DF
    for id_ in kinds_of_ids:
        citation_with_ids[id_] = None

    print('Total kind of Citation IDs: {}'.format(len(kinds_of_ids)))

    # Set the value of identifiers for each column, for e.g. DOI, ISBN etc.
    def set_citation_val(x):
        for item in x['ID_list']:
            citation_with_ids.at[x.name, item[0]] = item[1] if len(item) >= 2 else None

    # _ = citation_with_ids.progress_apply(lambda x: set_citation_val(x), axis=1)
    _ = citation_with_ids.apply(lambda x: set_citation_val(x), axis=1)

    print("CITATIONS_WITH_IDS ID_LIST", citation_with_ids['ID_list'])

    # Setting the labels
    citation_with_ids['actual_label'] = 'rest'

    is_doi  = 'DOI' in citation_with_ids
    is_pmc  = 'PMC' in citation_with_ids
    is_pmid = 'PMID' in citation_with_ids
    is_isbn = 'ISBN'in citation_with_ids

    # NK TODO No 'PMC' is in citations_with_ids
    if is_pmc:
        citation_with_ids.loc[~pd.isna(citation_with_ids['PMC']), ['actual_label']] = 'journal'
    if is_pmid:
        citation_with_ids.loc[~pd.isna(citation_with_ids['PMID']), ['actual_label']] = 'journal'

    only_doi = (
        ~pd.isna(citation_with_ids['DOI']) &
        pd.isna(citation_with_ids['PMC']) &
        pd.isna(citation_with_ids['PMID']) &
        pd.isna(citation_with_ids['ISBN'])
    )

    print("ONLY DOI:", only_doi)
    citation_with_ids.loc[only_doi, ['actual_label']] = 'journal'

    only_book = (
        ~pd.isna(citation_with_ids['ISBN']) &
        pd.isna(citation_with_ids['PMC']) &
        pd.isna(citation_with_ids['PMID']) &
        pd.isna(citation_with_ids['DOI'])
    )
    citation_with_ids.loc[only_book, ['actual_label']] = 'book'

    both = (
        ~pd.isna(citation_with_ids['ISBN']) &
        ~pd.isna(citation_with_ids['DOI']) &
        pd.isna(citation_with_ids['PMC']) &
        pd.isna(citation_with_ids['PMID'])
    )

    both_book_and_doi_journal = (both &
        citation_with_ids['type_of_citation'].isin(['cite journal', 'cite conference'])
    )
    citation_with_ids.loc[both_book_and_doi_journal, ['actual_label']] = 'journal'

    both_book_and_doi_book = (both &
        citation_with_ids['type_of_citation'].isin(['cite book', 'cite encyclopedia'])
    )
    citation_with_ids.loc[both_book_and_doi_book, ['actual_label']] = 'book'

    # Made the dataset which contains citations book and journal labeled
    citation_with_ids = citation_with_ids[citation_with_ids['actual_label'].isin(['book', 'journal'])]

    print("Content of citations_with_ids:")
    print(citation_with_ids.head())

    citation_with_ids = citation_with_ids[[
      'type_of_citation', 'citations', 'id', 'ref_index', 'sections',
      'total_words', 'neighboring_tags', 'actual_label', 'neighboring_words'
    ]]
    print('The total number of citations_with_ids: {}'.format(citation_with_ids.shape))

    citation_with_ids = citation_with_ids.set_index(['id', 'citations'])
    citation_with_ids = citation_with_ids[~citation_with_ids.index.duplicated(keep='first')]
    citation_with_ids = citation_with_ids.reset_index()

    print('IDs for book and journal which fulfil the criteria: {}'.format(citation_with_ids.shape[0]))

    # Removing the biases
    labels = ['doi', 'isbn', 'pmc', 'pmid', 'url', 'work', 'newspaper', 'website']
    for label in labels:
        # citation_with_ids['citations'] = citation_with_ids['citations'].progress_apply(
        citation_with_ids['citations'] = citation_with_ids['citations'].apply(
            lambda x: re.sub(label+'\s{0,10}=\s{0,10}([^|]+)', label+' = ', x))

    citation_with_ids['actual_label'].value_counts()
    citation_with_ids['actual_prob'] = citation_with_ids['actual_label']\
        .apply(lambda x: 0.45 if x == 'book' else 0.55)
        # .progress_apply(lambda x: 0.45 if x == 'book' else 0.55)

    n = min(int(len(citation_with_ids) / 2), 1000000)
    book_journal_features = citation_with_ids.sample(n=n, weights='actual_prob')
    print(book_journal_features['actual_label'].value_counts())
    book_journal_features.to_parquet(file_out)
    book_journal_features.to_csv(file_out_csv, sep='\t')
