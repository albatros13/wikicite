# -*- coding: utf-8 -*-
"""
Get newspaper citations based on the Top level domain from the URL feature of the citations.
Can also be used for extraction of entertainment/videos citations
"""

import tldextract
from pyspark.sql.functions import udf, col


def get_newspaper_citations(sql_context, file_in, file_out):
    print("Step 8: Getting newspaper citations...")

    sql_context.setConf('spark.sql.parquet.compression.codec', 'snappy')
    citations_separated = sql_context.read.parquet(file_in)
    citations_separated = citations_separated.where(col("URL").isNotNull())

    def get_top_domain(citation_url):
        ext = tldextract.extract(citation_url)
        return ext.domain

    top_domain_udf = udf(get_top_domain)
    citations_separated = citations_separated.withColumn('tld', top_domain_udf('URL'))

    newspapers = {'nytimes', 'bbc', 'washingtonpost', 'cnn', 'theguardian', 'huffingtonpost', 'indiatimes'}

    citations_separated = citations_separated.where(col("tld").isin(newspapers))
    citations_separated.write.mode('overwrite').parquet(file_out)
