from local import PROJECT_HOME
from get_data import get_data
from get_generic_tmpl import get_generic_tmpl
from get_citation_keys import get_citation_keys, get_citation_ids
from features.get_content import get_content
from features.extract_nlp_features import extract_nlp_features
from features.get_dataset_features import get_dataset_features
from features.filter_content import filter_content
from features.get_selected_features import get_selected_features
from get_book_journal_features import get_book_journal_features
from get_newspaper_citations import get_newspaper_citations
from get_test_data import get_test_data
from pyspark import SparkContext, SQLContext
from predict_citations import predict_citations

ext = "_xh"
INPUT_DATA = PROJECT_HOME + 'data/dumps/xhwiki-20221001-pages-articles-multistream.xml.bz2'

sc = SparkContext()
sql_context = SQLContext(sc)

# files
CITATIONS = PROJECT_HOME + 'data/content/citations' + ext + '.parquet'
CITATIONS_GENERIC = PROJECT_HOME + 'data/content/generic_citations' + ext + '.parquet'
CITATIONS_SEPARATED = PROJECT_HOME + 'data/content/citations_separated' + ext + '.parquet'

CITATIONS_CONTENT = PROJECT_HOME + 'data/content/citations_content' + ext + '.parquet'
CITATIONS_IDS = PROJECT_HOME + 'data/content/citations_ids' + ext + '.parquet'

BASE_FEATURES = PROJECT_HOME + 'data/features/base_features_complete' + ext + '.parquet'
CITATIONS_FEATURES = PROJECT_HOME + 'data/features/citations_features' + ext + '.parquet'

BOOK_JOURNAL_CITATIONS = PROJECT_HOME + 'data/features/book_journal_citations' + ext + '.parquet'
BOOK_JOURNAL_CITATIONS_CSV = PROJECT_HOME + 'data/features/book_journal_citations' + ext + '.csv'

NEWSPAPER_CITATIONS = PROJECT_HOME + 'data/features/newspaper_citations' + ext + '.parquet'
NEWSPAPER_FEATURES = PROJECT_HOME + 'data/features/newspaper_citation_features'+ext+'.parquet'
NEWSPAPER_FEATURES_CSV = PROJECT_HOME + 'data/features/newspaper_citation_features'+ext+'.csv'

ENTERTAINMENT_TITLES_DATA = PROJECT_HOME + 'data/content/entertainment_titles' + ext + '.csv'
ENTERTAINMENT_CITATIONS = PROJECT_HOME+'data/content/entertainment_citations'+ext+'.parquet'
# ENTERTAINMENT_FEATURES = PROJECT_HOME + 'data/features/entertainment_features'+ext+'.parquet'
# ENTERTAINMENT_FEATURES_CSV = PROJECT_HOME + 'data/features/entertainment_features'+ext+'.csv'

TEST_RANDOM_PAGES = PROJECT_HOME + 'data/content/random_pages' + ext + '.parquet'
TEST_RANDOM_PAGES_CSV = PROJECT_HOME + 'data/content/random_pages' + ext + '.csv'

start_from_step = 100

# 1
if start_from_step <= 1:
    print("Step 1: Getting citations from XML dump...")
    get_data(sql_context, INPUT_DATA, CITATIONS)

# 2
if start_from_step <= 2:
    print("Step 2: Converting citations to generic template...")
    get_generic_tmpl(sql_context, CITATIONS, CITATIONS_GENERIC)

# 3
if start_from_step <= 3:
    print("Step 3: Creating citation dictionary...")
    get_citation_keys(sql_context, CITATIONS_GENERIC, CITATIONS_SEPARATED)
    get_citation_ids(sql_context, CITATIONS_SEPARATED, CITATIONS_IDS)

# FEATURES

# 4
if start_from_step <= 4:
    print("Step 4: Getting content from XML dump...")
    get_content(sql_context, INPUT_DATA, CITATIONS_CONTENT)

# 5
if start_from_step <= 5:
    print("Step 5: Extracting NLP features...")
    extract_nlp_features(sql_context, CITATIONS_CONTENT, BASE_FEATURES)

# 6
if start_from_step <= 6:
    print("Step 6: Getting dataset features...")
    get_dataset_features(sql_context, BASE_FEATURES, CITATIONS_SEPARATED, CITATIONS_FEATURES)

start_from_step = 100

# 7
if start_from_step <= 7:
    print("Step 7: Predicting citations...")
    predict_citations(sql_context, BASE_FEATURES, CITATIONS_SEPARATED, CITATIONS_FEATURES)

# BOOKS AND JOURNALS

# 10

if start_from_step <= 10:
    print("Step 10: Getting book and journal citations...")
    get_book_journal_features(CITATIONS_FEATURES, BOOK_JOURNAL_CITATIONS, BOOK_JOURNAL_CITATIONS_CSV)

# NEWSPAPERS

start_from_step = 21

# 20
if start_from_step <= 20:
    print("Step 20: Getting newspaper citations...")
    get_newspaper_citations(sql_context, CITATIONS_SEPARATED, NEWSPAPER_CITATIONS)
    # TODO get data with required column names (like in book_journal_features
    #     'type_of_citation', 'citations', 'id', 'ref_index', 'sections',
    #     'total_words', 'neighboring_tags', 'actual_label', 'neighboring_words'

# 21 this seems to do the same as general step 6? Also, faulty - no citations are left
# if start_from_step <= 21:
#     print("Step 21: Getting newspaper features...")
#     get_selected_features(sql_context, BASE_FEATURES, NEWSPAPER_CITATIONS, NEWSPAPER_FEATURES)


# ENTERTAINMENT

# 30
# if start_from_step <= 30:
#     print("Step 30: Filtering entertainment citations...")
#     filter_content(CITATIONS_CONTENT, ENTERTAINMENT_TITLES_DATA, ENTERTAINMENT_CITATIONS)
#     # TODO remove when ENTERTAINMENT_TITLES_DATA is collected
#     ENTERTAINMENT_CITATIONS = CITATIONS_SEPARATED

# if start_from_step <= 31:
#     print("Step 31: Filtering entertainment citations...")
#     get_selected_features(sql_context, BASE_FEATURES, ENTERTAINMENT_CITATIONS,
#             ENTERTAINMENT_FEATURES, ENTERTAINMENT_FEATURES_CSV)

# TESTING

# TODO skip testing for now
start_from_step = 100

# 40
if start_from_step <= 40:
    print("Step 40: Sampling test data...")
    get_test_data(sql_context, INPUT_DATA, TEST_RANDOM_PAGES, TEST_RANDOM_PAGES_CSV)