# -*- coding: utf-8 -*-
"""
Get newspaper citations based on the Top level domain from the URL feature of the citations.
Can also be used for extraction of entertainment/videos citations
"""

import tldextract
from pyspark.sql.functions import udf, col, regexp_replace


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

    # newspapers = {'nytimes', 'bbc', 'washingtonpost', 'cnn', 'theguardian', 'huffingtonpost', 'indiatimes',
    #               'independent', 'dailymail', 'telegraph', 'timesonline', 'reuters'}

    # NK distinct set of newpapers from 'cite news' dataset
    # citations_separated.select("tld").distinct().write.format('com.databricks.spark.csv').save('./data/newspaper_domains.csv')

    newspapers = {'globo', 'signonsandiego', 'terra ', 'bbc', 'uol', 'cnn', 'espncricinfo', 'news18', 'msn',
                  'theguardian', 'washingtonpost', 'guardian', 'telegraph', 'reuters',
                  'cbc', 'mg', 'dailymail', 'youthvillage', 'afternoonexpress', 'usatoday', 'goal', 'timesonline',
                  'tvsa', 'slashdot', 'mtvbase', 'dailysun', 'chicagotribune', 'news24', 'skysports', 'smh', 'billboard',
                  'fifa', 'nytimes', 'iol', 'rsssf', 'independent', 'nupedia', 'yahoo'}

    news_templates = {'cite news'}

    citations_separated = citations_separated.where(col("type_of_citation").isin(news_templates) | col("tld").isin(newspapers))
    # NK I added elimination of " to match citations with features
    citations_separated = citations_separated.withColumn('citations', regexp_replace('citations', '"', ''))
    citations_separated.write.mode('overwrite').parquet(file_out)
