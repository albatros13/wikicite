"""
Get random Wikicode formatted citations based on the content of the Wikipedia page for testing purposes.
"""

def get_test_data(sql_context, file_in, file_out, file_out_csv=None):
    print("Testing Step 1: Sampling test data...")
    wiki = sql_context.read.format('com.databricks.spark.xml').options(rowTag='page').load(file_in)
    pages = wiki.where('ns = 0').where('redirect is null')

    # Get only ID, title, revision text's value which we are interested in
    # NK replaced #VALUE -> _VALUE
    pages = pages['id', 'title', 'revision.text._VALUE', 'revision.id', 'revision.parentid']
    pages = pages.toDF('id', 'title', 'content', 'r_id', 'r_parentid')

    random_pages = sql_context.createDataFrame(pages.rdd.takeSample(False, 5, seed=0))
    random_pages.write.mode('overwrite').parquet(file_out)
    if file_out_csv:
       random_pages.write.format('com.databricks.spark.csv').save(file_out_csv)
