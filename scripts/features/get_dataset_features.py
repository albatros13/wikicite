from pyspark.sql.functions import col


def get_dataset_features(sql_context, file_in1, file_in2, file_out):

    sql_context.setConf('spark.sql.parquet.compression.codec', 'snappy')
    base_features = sql_context.read.parquet(file_in1)
    dataset_citations = sql_context.read.parquet(file_in2)

    base_features = base_features.select(
        col('id').alias('page_id'),
        col('citations_features._1').alias('retrieved_citation'),
        col('citations_features._2').alias('ref_index'),
        col('citations_features._3').alias('total_words'),
        col('citations_features._4._1').alias('neighboring_words'),
        col('citations_features._4._2').alias('neighboring_tags')
    )

    filtered = dataset_citations.join(
        base_features,
        (base_features.page_id == dataset_citations.id),
        how='inner'
    )

    filtered = filtered.select(
        'id',
        'citations',
        'page_title',
        'ID_list',
        'type_of_citation',
        'page_id',
        'sections',
        'retrieved_citation',
        'ref_index',
        'total_words',
        'neighboring_words',
        'neighboring_tags'
    )

    # Drop the column since there are 2 columns with citations
    filtered = filtered.drop('retrieved_citation')
    filtered.write.mode('overwrite').parquet(file_out)



