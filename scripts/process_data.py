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
ENTERTAINMENT_FEATURES = PROJECT_HOME + 'data/features/entertainment_features'+ext+'.parquet'

# # 1
# get_data(sql_context, INPUT_DATA, CITATIONS)
#
# # 2
# get_generic_tmpl(sql_context, CITATIONS, CITATIONS_GENERIC)
#
# # 3
# get_citation_keys(sql_context, CITATIONS_GENERIC, CITATIONS_SEPARATED)
# get_citation_ids(sql_context, CITATIONS_SEPARATED, CITATIONS_IDS)
#
# # 4
# get_content(sql_context, INPUT_DATA, CITATIONS_CONTENT)
#
# # 5
# extract_nlp_features(sql_context, CITATIONS_CONTENT, BASE_FEATURES)
#
# # 6
# get_dataset_features(sql_context, BASE_FEATURES, CITATIONS_SEPARATED, CITATIONS_FEATURES)

# 7
# get_book_journal_features(CITATIONS_FEATURES, BOOK_JOURNAL_CITATIONS, BOOK_JOURNAL_CITATIONS_CSV)

# 8
get_newspaper_citations(sql_context, CITATIONS_SEPARATED, NEWSPAPER_CITATIONS)

# 9
get_selected_features(sql_context, BASE_FEATURES, NEWSPAPER_CITATIONS, NEWSPAPER_FEATURES)

# 10
# filter_content(CITATIONS_CONTENT, ENTERTAINMENT_TITLES_DATA, ENTERTAINMENT_CITATIONS)
#
# # 11
# get_selected_features(sql_context, BASE_FEATURES, CITATIONS_SEPARATED, CITATIONS_FEATURES)
#
# # MAIN
# predict_citations(PROJECT_HOME, ext)
