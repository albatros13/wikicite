from pyspark import SparkContext, SQLContext


def get_content(sql_context, file_in, file_out):
    print("Step 4: Getting content from XML dump...")

    sql_context.setConf('spark.sql.parquet.compression.codec', 'snappy')
    wiki = sql_context.read.format('com.databricks.spark.xml').options(rowTag='page').load(file_in)
    pages = wiki.where('ns = 0').where('redirect is null')

    # Get only ID, title, revision text's value which we are interested in
    pages = pages['id', 'title', 'revision.text']
    pages = pages.toDF('id', 'page_title', 'content')
    pages.write.mode('overwrite').parquet(file_out)

