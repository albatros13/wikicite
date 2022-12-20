from local import PROJECT_HOME
from get_test_data import get_test_data
from pyspark import SparkContext, SQLContext

ext = "_xh"
INPUT_DATA = PROJECT_HOME + 'data/dumps/xhwiki-20221001-pages-articles-multistream.xml.bz2'

sc = SparkContext()
sql_context = SQLContext(sc)

# TESTING
TEST_RANDOM_PAGES = PROJECT_HOME + 'data/content/random_pages' + ext + '.parquet'
TEST_RANDOM_PAGES_CSV = PROJECT_HOME + 'data/content/random_pages' + ext + '.csv'

# 1
get_test_data(sql_context, INPUT_DATA, TEST_RANDOM_PAGES, TEST_RANDOM_PAGES_CSV)