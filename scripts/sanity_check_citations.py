import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scripts.const import CITATION_TEMPLATES
from pyspark.sql.functions import when, isnan, col, count
from lookup.run_apis import run_google_book_get_info, run_crossref_get_info
from fuzzywuzzy import fuzz
import findspark
from pyspark import SparkContext, SQLContext
import warnings

findspark.init()
warnings.filterwarnings("ignore")

sc = SparkContext()
sql_context = SQLContext(sc)

# ext = "xh_"
ext = "en_"

PROJECT_HOME = 'c:///users/natal/PycharmProjects/cite-classifications-wiki/'
TOP_300_TEMPLATES = PROJECT_HOME + 'data/top300_templates.csv'

CITATIONS_SEPARATED = PROJECT_HOME + 'data/content/{}citations_separated.parquet'.format(ext)
CITATIONS_IDS = PROJECT_HOME + 'data/content/{}citations_ids.parquet'.format(ext)
CITATIONS_WITH_IDS = PROJECT_HOME + 'data/content/{}citations_with_ids.csv'.format(ext)

all_citations = sql_context.read.parquet(CITATIONS_SEPARATED)
citation_count = all_citations.groupby('type_of_citation').count().toPandas()

print("Citation types count:", citation_count)
print("Unique citation types:", citation_count['type_of_citation'].unique())

number_of_values_with_null = all_citations.select([
    count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in all_citations.columns]).toPandas()

null_over_sixty_percent = ((number_of_values_with_null / citation_count['count'].sum()) * 100).iloc[0] > 60
null_over_sixty_percent[null_over_sixty_percent == True]

top300_templates = pd.read_csv(TOP_300_TEMPLATES)
# Only consider the templates which can be parsed by mwparserfromhell
parseable_template_count = top300_templates.loc[top300_templates['template'].isin(CITATION_TEMPLATES)]

merged_counts = pd.merge(
    parseable_template_count, citation_count,
    left_on='template', right_on='type_of_citation', how='inner'
).drop('template', axis=1)

merged_counts.columns = ['dlab_count', 'type_of_citation', 'curated_count']
merged_counts['curated_count'].sum()
merged_counts['bigger_than'] = merged_counts['curated_count'] - merged_counts['dlab_count']

print("Merged counts")
print(merged_counts)

# citation_with_ids = pd.read_parquet(CITATIONS_IDS, engine='pyarrow')
citation_with_ids = pd.read_parquet(CITATIONS_IDS)

print("Citations read")

print(citation_with_ids.shape)
total_citations = citation_with_ids.shape[0]

print("Citations with IDs")
citation_with_ids.head()

citation_with_ids.columns = [
    'id', 'page_title', 'citation', 'id_list', 'authors',
    'citation_title', 'citation_type', 'publisher_name', 'sections'
]

# Percentage of values present for title of page, title of citation and authors
print((citation_with_ids.count() * 100) / total_citations)
print(citation_with_ids.groupby('citation_type').size())

# Formulate a structure for the ID_List in which we can do something meaningful
citation_with_ids['id_list'] = citation_with_ids['id_list'].apply(
    lambda x: list(item.split('=') for item in x.replace('{','').replace('}','').replace(' ', '').split(','))
)

# Get the kinds of ids associated with each tuple
kinds_of_ids = set()
def update_ids(x):
    for item in x:
        kinds_of_ids.add(item[0])

_ = citation_with_ids['id_list'].apply(lambda x: update_ids(x))

# Add the columns with NoneType in the previous DF
for id_ in kinds_of_ids:
    citation_with_ids[id_] = None

print('Total kind of Citation IDs: {}'.format(len(kinds_of_ids)))


# Set the value of identifiers for each column, for e.g. DOI, ISBN etc.
def set_citation_val(x):
    for item in x['id_list']:
        citation_with_ids.at[x.name, item[0]] = item[1] if len(item) >= 2 else None


_ = citation_with_ids.apply(lambda x: set_citation_val(x), axis=1)

citation_with_ids.head()

# Save the file in the Pandas format
citation_with_ids.to_csv(CITATIONS_WITH_IDS)

# Let's take a few samples and query the crossref and Google books API for DOI and ISBN respectively.
mask_isbn_or_doi = citation_with_ids['DOI'].notnull() | citation_with_ids['ISBN'].notnull()
citation_with_isbn_or_doi = citation_with_ids[mask_isbn_or_doi][['id', 'citation_title', 'ISBN', 'DOI', 'authors']]
citation_with_isbn_or_doi = citation_with_isbn_or_doi.sample(n=100)
print(citation_with_isbn_or_doi.head())

citation_with_isbn_or_doi['retrieved_title'] = [[] for i in citation_with_isbn_or_doi.index]
citation_with_isbn_or_doi['retrieved_author'] = [[] for i in citation_with_isbn_or_doi.index]
citation_with_isbn_or_doi['api_type'] = ['' for i in citation_with_isbn_or_doi.index]

# Crossref is great for DOI, but does not return a lot of information for ISBN.
# Google books is better for ISBN but it limits the amount of requests one could send
# so that's why we are testing it on a smaller sample case.

for i in range(len(citation_with_isbn_or_doi)):
    title = []
    author = []
    row = citation_with_isbn_or_doi.iloc[i] # Get the particular row
    if row['DOI']:
        result_crossref = run_crossref_get_info(doi=row['DOI'])
        citation_with_isbn_or_doi.iloc[i, 7] = 'Crossref'

        if result_crossref.status_code != 200:
            title.append('No title mentioned')
            author.append('No authors mentioned')
            continue

        crossref_message = result_crossref.json()['message']

        if 'title' in crossref_message:
            title.extend(crossref_message['title'])
        else:
            title.append('No title mentioned')

        if 'author' in crossref_message:
            author.extend([
                a.get('given', '') + ' ' + a.get('family', '')
                for a in crossref_message['author']
            ])
        else:
            author.append('No authors mentioned')

    if not row['DOI'] and row['ISBN']:
        isbn = row['ISBN'].replace('-', '')
        result_google = run_google_book_get_info(isbn=isbn).json()
        citation_with_isbn_or_doi.iloc[i, 7] = 'Google'

        if 'items' not in result_google:
            row['retrieved_title'] = 'No title mentioned'
            row['retrieved_author'] = 'No authors mentioned'
            continue

        for item in result_google['items']:
            title.append(item['volumeInfo'].get('title', 'No title mentioned'))
            author.extend(item['volumeInfo'].get('authors', ['No authors mentioned']))

    if i % 10 == 0:
        print('Done with {} citations'.format(i))

    citation_with_isbn_or_doi.iloc[i]['retrieved_title'].extend(title)
    citation_with_isbn_or_doi.iloc[i]['retrieved_author'].extend(author)

# citation_with_isbn_or_doi
# Doing analysis of the Google and Crossref API - as to how many authors and title are equal?

total_google_samples = len(citation_with_isbn_or_doi[citation_with_isbn_or_doi['api_type'] == 'Google'])
total_crossref_samples = len(citation_with_isbn_or_doi[citation_with_isbn_or_doi['api_type'] == 'Crossref'])

print('Google Samples: {}\nCrossref Samples: {}'.format(total_google_samples, total_crossref_samples))

# Lets perform some API robustness test on titles..
# Using Fuzzy String Matching to get an approximate matching of string instead of actual one


def get_ratio(row, col1, col2):
    actual_ = row[col1] if row[col1] else 'No title'
    retrieved_ = row[col2][0] if len(row[col2]) >= 1 else 'No retrieved title'
    return fuzz.token_set_ratio(actual_, retrieved_)


# As you can see Crossref is more precise with its results and its API is more robust than Google Books since
# Google Books is more like a search engine for books and returns more broader results.
# Also, Crossref is a specialist API and hence results are more specific.
# Some of the edge cases which are not addressed is sometimes the title retrieved are in another language,
# but these cases are far and less.

citation_with_isbn_or_doi['title_percent_match'] = citation_with_isbn_or_doi.apply(
    get_ratio, args=('citation_title', 'retrieved_title'), axis=1)
citation_with_isbn_or_doi[['api_type', 'title_percent_match']].groupby('api_type').mean()


# Lets now apply the robustness test on authors..


# Preprocess authors so that we disappear the 'last=' and 'first=' phrase and convert them into list
def preprocess_authors(row):
    authors = (
        'No authors' if isinstance(row['authors'], float) or not row['authors']
        else row['authors'].split('}, {')
    )
    for ch in [']', '[', '{', '}', 'first=', 'last=', ',', 'link=']:
        authors = [i.replace(ch, '') for i in authors]
    return len(authors), ', '.join(authors)


citation_temp_authors = citation_with_isbn_or_doi[['authors', 'retrieved_author', 'api_type']]

# Get the length of the number of author and the length of the number of retrieved authors
citation_temp_authors['len_authors'], citation_temp_authors['joined_authors'] = zip(*citation_temp_authors.apply(preprocess_authors, axis=1))
citation_temp_authors['len_retrieved_author'] = citation_temp_authors['retrieved_author'].apply(lambda x: len(x))
citation_temp_authors['retrieved_author'] = citation_temp_authors['retrieved_author'].apply(lambda x: ', '.join(x))

(citation_temp_authors['len_authors'] <= citation_temp_authors['len_retrieved_author']).value_counts()

citation_temp_authors['author_percent_match'] = citation_temp_authors.apply(
    get_ratio, args=('joined_authors', 'retrieved_author'), axis=1)
citation_temp_authors[['api_type', 'author_percent_match']].groupby('api_type').mean()

# NK TODO cross-validate Google API return against title? (Levenshtein distance)

# How do the identifiers appear with each other?
# Do we have many citations with two identifiers?

identifiers_existing = citation_with_ids[['DOI', 'ISBN', 'ISSN', 'PMC', 'PMID']].notnull()

identifiers_existing.head()
all_columns = identifiers_existing.columns
frequency_citation = dict()


def get_frequency_of_identifiers_appearing(x):
    available_citation_types = tuple([column for column in all_columns if x[column]])
    frequency_citation.setdefault(available_citation_types, 0)
    frequency_citation[available_citation_types] += 1


_ = identifiers_existing.apply(lambda x: get_frequency_of_identifiers_appearing(x), axis=1)


# Make a graph of the frequency distribution calculated above
names = list(frequency_citation.keys())
values = list(frequency_citation.values())

plt.figure(111)
fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = (12,5)
plt.xticks(rotation=90)
plt.bar(range(len(frequency_citation)),values,tick_label=names)
fig.tight_layout()
plt.show()

##########################################################################

# NK TODO where is in this file?
# NK Where do I get it?
CITATION_WITH_IDENTIFIERS = '../Citations_with_Identifiers/enwiki.tsv.tar.gz'
def check_wiki_en_identifiers():
    # TODO NK where does 'wikipedia dataset with identifiers' come from?

    # Loading the wikipedia dataset with identifiers
    wiki_en_identifiers = pd.read_csv(CITATION_WITH_IDENTIFIERS, compression='gzip', sep='\t')
    wiki_en_identifiers.head(5)

    print('Total citation identifiers for Wikipedia dump: {}'.format(wiki_en_identifiers.shape[0]))
    wiki_en_identifiers['type'].unique() # Labels which have unique IDSs
    # Remove the one with the NaN value
    wiki_en_identifiers = wiki_en_identifiers[wiki_en_identifiers['type'].notnull()]

    # Adding a boolean to check if the citation is in other dataset - to
    wiki_en_identifiers['is_in_other_dataset'] = False
    wiki_en_identifiers['page_title'].nunique()
    wiki_en_identifiers['id'].nunique()

    # Revision Analysis
    curated_title_id = citation_with_ids[['page_', 'r_id', 'r_parentid']]
    curated_title_id.head()

    # As we can see, many parent ids in our dataset are not present in the citation with identifiers dataset
    # which should be kept in mind for further analysis and can be classified as a reason that we might get less citations.

    r_parentid_which_are_present = curated_title_id['r_parentid'].isin(wiki_en_identifiers['rev_id'])
    total_number_of_r_parentid_in_wiki = np.sum(r_parentid_which_are_present)
    print(curated_title_id.shape[0], wiki_en_identifiers['rev_id'].shape[0], total_number_of_r_parentid_in_wiki)

    # Comparing the two datasets
    # The gap exists between the two datasets (3.74 mil, 3.14 mil) of about 600,000 because we are looking only at
    # certain citation formats which can be parsed by the `mwparserfromhell`. But still we have got 90% of the citation
    # data by looking at just mere numbers. The 10% deficit is because of the dataset used by wiki identifiers is for
    # revision where we are using a dataset relating to a particular date.

    gap = wiki_en_identifiers.shape[0] - total_citations
    print('The total gap between between total number of wikipedias citations and our citations: {}'.format(gap))


    def get_citations_specific_to_type(wiki_type, curated_type):
        type_wiki_identifiers = wiki_en_identifiers[wiki_en_identifiers['type'] == wiki_type]
        type_citations_curated = citation_with_ids[citation_with_ids[curated_type].notnull()]

        # Just considering the unique ones since they are a lot of duplicated DOIs
        # Maybe one citation is cited in many different pages
        number_of_identifiers_wiki = type_wiki_identifiers['id'].shape[0]
        number_of_identifiers_curated = type_citations_curated['DOI'].shape[0]
        print('The total number of unique {} wiki identifiers: {}'.format(wiki_type, number_of_identifiers_wiki))
        print('The total number of unique {} curated identifiers: {}'.format(curated_type, number_of_identifiers_curated))
        print('\nThe difference between wiki and curated is: {}'.format(
            number_of_identifiers_wiki - number_of_identifiers_curated)
        )
        return type_wiki_identifiers, type_citations_curated


    # How many DOI identifiers are common?

    doi_wiki_identifiers, doi_citations_curated = get_citations_specific_to_type('doi', 'DOI')

    # Check if curated DOIs are contained in the already obtained dataset from Wikipedia
    doi_which_are_present = doi_wiki_identifiers['id'].isin(doi_citations_curated['DOI'])
    total_number_of_doi_identifiers_in_wiki = np.sum(doi_which_are_present)
    wiki_en_identifiers['is_in_other_dataset'].loc[doi_which_are_present.index] = doi_which_are_present.values

    print(
        'Stats:\nTotal Curated: {} \nTotal Wiki:{} \nCurated which are in Wiki: {} \nGap: {} -> Wiki which are not identified: {}'.format(
            doi_citations_curated.shape[0],
            doi_wiki_identifiers.shape[0],
            total_number_of_doi_identifiers_in_wiki,
            doi_citations_curated.shape[0] - total_number_of_doi_identifiers_in_wiki,
            doi_which_are_present[~doi_which_are_present].shape[0]
        )
    )

    # How many ISBN (also ISSN) identifiers are common?

    # * ISBNs are Book Numbers. They can be assigned to monographic publications, such as books, e-books and audiobooks.
    # * ISMNs are Music Numbers. They can be assigned to notated music (scores and sheet music) whether published in print,
    # online or in other media.
    # * ISSNs are Serial Numbers. They can be assigned to periodical publications, such as magazines and journals.

    isbn_wiki_identifiers, isbn_citations_curated = get_citations_specific_to_type('isbn', 'ISBN')

    # Trying to normalize all the ISBN (also need to do for ISSN)
    # * So if '00-11-223344' it becomes '0011223344'

    # Check if the wikipedia citation identifiers does not have hyphens
    np.sum(isbn_wiki_identifiers['id'].apply(lambda x: '-' in x))
    isbn_citations_curated['ISBN'] = isbn_citations_curated['ISBN'].apply(lambda x: x.replace('-', ''))

    # Check if curated DOIs are contained in the already obtained dataset from Wikipedia

    isbn_which_are_present = isbn_wiki_identifiers['id'].isin(isbn_citations_curated['ISBN'])
    total_number_of_isbn_identifiers_in_wiki = np.sum(isbn_which_are_present)
    wiki_en_identifiers['is_in_other_dataset'].loc[isbn_which_are_present.index] = isbn_which_are_present.values

    print(
        'Stats:\nTotal Curated: {} \nTotal Wiki:{} \nCurated which are in Wiki: {} \nGap: {} -> Wiki which are not identified: {}'.format(
            isbn_citations_curated.shape[0],
            isbn_wiki_identifiers.shape[0],
            total_number_of_isbn_identifiers_in_wiki,
            isbn_citations_curated.shape[0] - total_number_of_isbn_identifiers_in_wiki,
            isbn_which_are_present[~isbn_which_are_present].shape[0]
        )
    )

    # Now time for ISSN...

    # But the stats for this does not matter!!!
    # because the hypothesis is that ISSN is contained inside ISBN - but only some of them do!
    isbn_wiki_identifiers, issn_citations_curated = get_citations_specific_to_type('isbn', 'ISSN')

    # Normalizing it again like ISBN
    issn_citations_curated['ISSN'] = issn_citations_curated['ISSN'].apply(lambda x: x.replace('-', ''))

    issn_which_are_present = issn_citations_curated['ISSN'].isin(isbn_wiki_identifiers['id'])
    total_number_of_issn_identifiers_in_wiki = np.sum(issn_which_are_present)
    wiki_en_identifiers['is_in_other_dataset'].loc[issn_which_are_present.index] = issn_which_are_present.values

    print(
        'Stats:\nTotal Curated: {} \nCurated which are in Wiki: {} \nGap: {} -> Wiki which are not identified: {}'.format(
            issn_citations_curated.shape[0],
            total_number_of_issn_identifiers_in_wiki,
            issn_citations_curated.shape[0] - total_number_of_issn_identifiers_in_wiki,
            issn_which_are_present[~issn_which_are_present].shape[0]
        )
    )

    # What we can see is that ISSN exists in our `curated` dataset and only some of them of these are contained in
    # the existing `wikipedia dataset`. Most of them do not exist and hence the hypothesis is potentially not
    # correct. Also, some of these can be counter examples since they are magazines and music volumes which are
    # not scientific in nature.

    # How many PMID identifiers are common?

    pmid_wiki_identifiers, pmid_citations_curated = get_citations_specific_to_type('pmid', 'PMID')

    # Check if curated PMIDs are contained in the already obtained dataset from Wikipedia
    pmid_which_are_present = pmid_citations_curated['PMID'].isin(pmid_wiki_identifiers['id'])
    total_number_of_pmid_identifiers_in_wiki = np.sum(pmid_which_are_present)
    wiki_en_identifiers['is_in_other_dataset'].loc[pmid_which_are_present.index] = pmid_which_are_present.values

    print(
        'Stats:\nTotal Curated: {} \nTotal Wiki:{} \nCurated which are in Wiki: {} \nGap: {} -> Wiki which are not identified: {}'.format(
            pmid_citations_curated.shape[0],
            pmid_wiki_identifiers.shape[0],
            total_number_of_pmid_identifiers_in_wiki,
            pmid_citations_curated.shape[0] - total_number_of_pmid_identifiers_in_wiki,
            pmid_which_are_present[~pmid_which_are_present].shape[0]
        )
    )

    # How many PMC identifiers are common?
    pmc_wiki_identifiers, pmc_citations_curated = get_citations_specific_to_type('pmc', 'PMC')

    # Check if curated PMCs are contained in the already obtained dataset from Wikipedia
    pmc_which_are_present = pmc_citations_curated['PMC'].isin(pmc_wiki_identifiers['id'])
    total_number_of_pmc_identifiers_in_wiki = np.sum(pmc_which_are_present)
    wiki_en_identifiers['is_in_other_dataset'].loc[pmc_which_are_present.index] = pmc_which_are_present.values

    print(
        'Stats:\nTotal Curated: {} \nTotal Wiki:{} \nCurated which are in Wiki: {} \nGap: {} -> Wiki which are not identified: {}'.format(
            pmc_citations_curated.shape[0],
            pmc_wiki_identifiers.shape[0],
            total_number_of_pmc_identifiers_in_wiki,
            pmc_citations_curated.shape[0] - total_number_of_pmc_identifiers_in_wiki,
            pmc_which_are_present[~pmc_which_are_present].shape[0]
        )
    )

    # How many ArXiV identifiers are common?
    arxiv_wiki_identifiers, arxiv_citations_curated = get_citations_specific_to_type('arxiv', 'ARXIV')

    # Check if curated PMCs are contained in the already obtained dataset from Wikipedia
    arxiv_which_are_present = arxiv_citations_curated['ARXIV'].isin(arxiv_wiki_identifiers['id'])
    total_number_of_arxiv_identifiers_in_wiki = np.sum(arxiv_which_are_present)
    wiki_en_identifiers['is_in_other_dataset'].loc[arxiv_which_are_present.index] = arxiv_which_are_present.values

    print(
        'Stats:\nTotal Curated: {} \nTotal Wiki:{} \nCurated which are in Wiki: {} \nGap: {} -> Wiki which are not identified: {}'.format(
            arxiv_citations_curated.shape[0],
            arxiv_wiki_identifiers.shape[0],
            total_number_of_arxiv_identifiers_in_wiki,
            arxiv_citations_curated.shape[0] - total_number_of_arxiv_identifiers_in_wiki,
            arxiv_which_are_present[~arxiv_which_are_present].shape[0]
        )
    )

# check_wiki_en_identifiers()