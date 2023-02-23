from get_data import get_data
from get_generic_tmpl import get_generic_tmpl, only_with_ids
from features.extract_nlp_features import extract_nlp_features
from features.get_dataset_features import get_dataset_features
from features.get_selected_features import get_selected_features
from get_book_journal_features import get_book_journal_features, filter_with_ids
from get_newspaper_citations import get_newspaper_citations
from pyspark import SparkContext, SQLContext
import os
import sys
import findspark

os.environ["PYSPARK_SUBMIT_ARGS"] = " --packages com.databricks:spark-xml_2.12:0.15.0 pyspark-shell"
os.environ["JAVA_HOME"] = "C:/Program Files/Java/jdk-19/"
findspark.init()

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

print("Starting pyspark...")
sc = SparkContext.getOrCreate()
sql_context = SQLContext(sc)
print("Pyspark started...")

PROJECT_HOME = 'c:///users/natal/PycharmProjects/cite-classifications-wiki/'

# ext = "xh_"
# INPUT_DATA = PROJECT_HOME + 'data/dumps/xhwiki-20221001-pages-articles-multistream.xml.bz2'

ext = "en_"
INPUT_DATA = PROJECT_HOME + 'data/dumps/enwiki-20221201-pages-articles-multistream1.xml-p1p41242.bz2'

# ext = "_ro"
# INPUT_DATA = PROJECT_HOME + 'data/dumps/rowikinews-20230101-pages-articles-multistream.xml.bz2'

# ext = "_ar"
# INPUT_DATA = PROJECT_HOME + 'data/dumps/arwiki-20230101-pages-articles-multistream4.xml-p3982316p4045107.bz2'

# get_data
CITATIONS = PROJECT_HOME + 'data/content/{}citations.parquet'.format(ext)
CITATIONS_CONTENT = PROJECT_HOME + 'data/content/{}citations_content.parquet'.format(ext)

# extract_nlp_features
BASE_FEATURES = PROJECT_HOME + 'data/features/{}base_features_complete.parquet'.format(ext)

# get_generic_tmpl
CITATIONS_SEPARATED = PROJECT_HOME + 'data/content/{}citations_separated.parquet'.format(ext)

# get_dataset_features
CITATIONS_FEATURES = PROJECT_HOME + 'data/features/{}citations_features.parquet'.format(ext)
CITATIONS_FEATURES_IDS = PROJECT_HOME + 'data/features/{}citations_features_ids.parquet'.format(ext)

# label books and journals
BOOK_JOURNAL_CITATIONS = PROJECT_HOME + 'data/features/{}book_journal_citations.parquet'.format(ext)

# label newspapers (web?)
NEWSPAPER_CITATIONS = PROJECT_HOME + 'data/features/{}newspaper_citations.parquet'.format(ext)
NEWSPAPER_FEATURES = PROJECT_HOME + 'data/features/{}newspaper_citation_features.parquet'.format(ext)

# Optional
CITATIONS_IDS = PROJECT_HOME + 'data/content/{}citations_ids.parquet'.format(ext)

# Extract data

get_data(sql_context, INPUT_DATA, CITATIONS, CITATIONS_CONTENT, 50000)
extract_nlp_features(sql_context, CITATIONS_CONTENT, BASE_FEATURES)
get_generic_tmpl(sql_context, CITATIONS, CITATIONS_SEPARATED)
get_dataset_features(sql_context, BASE_FEATURES, CITATIONS_SEPARATED, CITATIONS_FEATURES)
filter_with_ids(sql_context, CITATIONS_FEATURES, CITATIONS_FEATURES_IDS)
get_book_journal_features(sql_context, CITATIONS_FEATURES_IDS, BOOK_JOURNAL_CITATIONS)
get_newspaper_citations(sql_context, CITATIONS_SEPARATED, NEWSPAPER_CITATIONS)
get_selected_features(sql_context, BASE_FEATURES, NEWSPAPER_CITATIONS, NEWSPAPER_FEATURES)

# Optional
# only_with_ids(sql_context, CITATIONS_SEPARATED, CITATIONS_IDS)

