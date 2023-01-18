from local import PROJECT_HOME
from get_data import get_data
from get_generic_tmpl import get_generic_tmpl, only_with_ids
from features.extract_nlp_features import extract_nlp_features
from features.get_dataset_features import get_dataset_features
from features.filter_content import filter_content
from features.get_selected_features import get_selected_features
from get_book_journal_features import get_book_journal_features
from get_newspaper_citations import get_newspaper_citations
from predict_citations import predict_citations
from pyspark import SparkContext, SQLContext
import os
import sys
import findspark

os.environ["PYSPARK_SUBMIT_ARGS"] = " --packages com.databricks:spark-xml_2.12:0.15.0 pyspark-shell"
os.environ["JAVA_HOME"] = "C:/Program Files/Java/jdk-19/"
findspark.init()

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


# NK tries this with spark-3.3.1-bin-hadoop3 - it fails on save, hadoop3 issue on Windows?
# os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages com.databricks:spark-xml_2.12-0.16.0 pyspark-shell"
# sc = SparkSession.builder.appName("WikiCite").config("spark.memory.offHeap.enabled","true")\
#     .config("spark.memory.offHeap.size","10g") \
#     .config("spark.jars", "PYSPARK_HOME/jars/spark-xml_2.12-0.9.0.jar") \
#     .getOrCreate()
#

print("Starting pyspark...")
sc = SparkContext()
sql_context = SQLContext(sc)
print("Pyspark started...")

ext = "_xh"
INPUT_DATA = PROJECT_HOME + 'data/dumps/xhwiki-20221001-pages-articles-multistream.xml.bz2'

# ext = "_en"
# INPUT_DATA = PROJECT_HOME + 'data/dumps/enwiki-20221201-pages-articles-multistream1.xml-p1p41242.bz2'

CITATIONS = PROJECT_HOME + 'data/content/citations' + ext + '.parquet'
CITATIONS_CONTENT = PROJECT_HOME + 'data/content/citations_content' + ext + '.parquet'

CITATIONS_SEPARATED = PROJECT_HOME + 'data/content/citations_separated' + ext + '.parquet'
CITATIONS_IDS = PROJECT_HOME + 'data/content/citations_ids' + ext + '.parquet'

BASE_FEATURES = PROJECT_HOME + 'data/features/base_features_complete' + ext + '.parquet'
CITATIONS_FEATURES = PROJECT_HOME + 'data/features/citations_features' + ext + '.parquet'

BOOK_JOURNAL_CITATIONS = PROJECT_HOME + 'data/features/book_journal_citations' + ext + '.parquet'
BOOK_JOURNAL_TEST = PROJECT_HOME + 'data/features/book_journal_test' + ext + '.parquet'

NEWSPAPER_CITATIONS = PROJECT_HOME + 'data/features/newspaper_citations' + ext + '.parquet'
NEWSPAPER_FEATURES = PROJECT_HOME + 'data/features/newspaper_citation_features' + ext + '.parquet'

# Extract data
# get_data(sql_context, INPUT_DATA, CITATIONS, CITATIONS_CONTENT, 50000)
# extract_nlp_features(sql_context, CITATIONS_CONTENT, BASE_FEATURES)
# get_generic_tmpl(sql_context, CITATIONS, CITATIONS_SEPARATED)

# get_dataset_features(sql_context, BASE_FEATURES, CITATIONS_SEPARATED, CITATIONS_FEATURES)

# get_book_journal_features(CITATIONS_FEATURES, BOOK_JOURNAL_CITATIONS, BOOK_JOURNAL_TEST)
# get_newspaper_citations(sql_context, CITATIONS_SEPARATED, NEWSPAPER_CITATIONS)
# get_selected_features(sql_context, BASE_FEATURES, NEWSPAPER_CITATIONS, NEWSPAPER_FEATURES)

# Train ML models before calling prediction!

# predict_citations(PROJECT_HOME, ext)

# Extra

# Citations with identifiers for sanity check
# only_with_ids(sql_context, CITATIONS_SEPARATED, CITATIONS_IDS)

