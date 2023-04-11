import unittest
from pyspark import SparkContext, SQLContext
import os
import findspark

os.environ["PYSPARK_SUBMIT_ARGS"] = " --packages com.databricks:spark-xml_2.12:0.15.0 pyspark-shell"
os.environ["JAVA_HOME"] = "C:/Program Files/Java/jdk-17/"
findspark.init()

sc = SparkContext()
sql_context = SQLContext(sc)


def get_test_data(sql_context, file_in, file_out):
    print("Testing Step 1: Sampling test data...")
    wiki = sql_context.read.format('com.databricks.spark.xml').options(rowTag='page').load(file_in)
    pages = wiki.where('ns = 0').where('redirect is null')

    pages = pages['id', 'title', 'revision.text._VALUE', 'revision.id', 'revision.parentid']
    pages = pages.toDF('id', 'title', 'content', 'r_id', 'r_parentid')

    random_pages = sql_context.createDataFrame(pages.rdd.takeSample(False, 5, seed=0))
    random_pages.write.mode('overwrite').parquet(file_out)


class TestCorrectNumberOfReferences(unittest.TestCase):

    def test_get_spark_data(self):
        PROJECT_HOME = 'c:///users/natal/PycharmProjects/cite-classifications-wiki/'
        ext = "xh_"
        INPUT_DATA = PROJECT_HOME + 'data/dumps/xhwiki-20221001-pages-articles-multistream.xml.bz2'
        TEST_RANDOM_PAGES = PROJECT_HOME + 'tests/data/{}random_pages.parquet'.format(ext)
        get_test_data(sql_context, INPUT_DATA, TEST_RANDOM_PAGES)
        # Assert that file exists
        self.assertTrue(os.path.exists(TEST_RANDOM_PAGES))


if __name__ == '__main__':
    unittest.main()
