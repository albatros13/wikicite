from pyspark.sql.functions import udf, col
from pyspark.sql.types import *


def get_selected_features(sql_context, file_in1, file_in2, file_out):
    print("Step 11: Getting newspaper features...")

    sql_context.setConf('spark.sql.parquet.compression.codec', 'snappy')
    features = sql_context.read.parquet(file_in1)
    features = features.withColumnRenamed('page_title', 'page_title_')

    features = features.select(
        col('citations_features._1').alias('retrieved_citation'),
        col('citations_features._2').alias('ref_index'),
        col('citations_features._3').alias('total_words'),
        col('citations_features._4._1').alias('neighboring_words'),
        col('citations_features._4._2').alias('neighboring_tags'),
    )

    selected_newspapers = sql_context.read.parquet(file_in2)

    def array_to_string(my_list):
       return '[' + ','.join([str(elem) for elem in my_list]) + ']'

    array_to_string_udf = udf(array_to_string,StringType())
    # print(selected_newspapers['citations'])
    # print("Features: ", features.head(20))

    results = features.join(selected_newspapers, features['retrieved_citation'] == selected_newspapers['citations'])
    results = results.withColumn('neighboring_words', array_to_string_udf(results["neighboring_words"]))
    results = results.withColumn('neighboring_tags', array_to_string_udf(results["neighboring_tags"]))

    results = results.drop('retrieved_citation')
    results.write.mode('overwrite').parquet(file_out)
