import re
from pyspark.sql.functions import col, explode, udf, lit
from pyspark.sql.types import BooleanType, ArrayType, StringType


def filter_with_ids(sql_context, file_in, file_out):
    print("Step 6a: Selecting citations with identifiers...")

    sql_context.setConf('spark.sql.parquet.compression.codec', 'snappy')
    citations_features = sql_context.read.parquet(file_in)

    print("Length full:", citations_features.count())
    print('The columns in the citations features are: {}'.format(citations_features.columns))

    citation_with_ids = citations_features.where(col("ID_list") != '')
    citation_with_ids = citation_with_ids.dropDuplicates()
    print("Length filtered:", citation_with_ids.count())
    citation_with_ids.write.mode('overwrite').parquet(file_out)


# Rewritten to perform necessary manipulations on Spark as opposed to Pandas dataframe as in the original script
def get_book_journal_features(sql_context, file_in, file_out):
    print("Step 6: Getting book and journal citations...")

    sql_context.setConf('spark.sql.parquet.compression.codec', 'snappy')
    citation_with_ids = sql_context.read.parquet(file_in)
    citation_with_ids = citation_with_ids.limit(50000)
    # df1 = citation_with_ids.limit(1000)
    # df2 = citation_with_ids.limit(2000)
    # citation_with_ids = df2.subtract(df1)

    parser = lambda x: list(re.split('=|:', item)
                            for item in
                            x.replace('{', '').replace('}', '').replace(' ', '').replace("'", "").split(','))
    udf_parser = udf(parser)
    citation_with_ids = citation_with_ids.withColumn("ID_list", udf_parser(citation_with_ids["ID_list"]))

    def update_ids(x):
        ids = []
        for item in x:
            ids.append(item[0])
        return ids

    udf_update_ids = udf(update_ids, ArrayType(StringType()))
    citation_with_ids = citation_with_ids.withColumn('kinds_of_ids', udf_update_ids(citation_with_ids['ID_list']))

    tmp = citation_with_ids.select("kinds_of_ids").distinct()
    tmp = tmp.withColumn("kinds_of_ids", explode("kinds_of_ids"))
    tmp = tmp.dropDuplicates(["kinds_of_ids"])
    kinds_of_ids = tmp.rdd.map(lambda x: x["kinds_of_ids"]).collect()
    print('Total kind of Citation IDs: {}'.format(kinds_of_ids))

    # kinds_of_ids = ['PMID', 'ARXIV', 'DOI', 'OL', 'PMC', 'LCCN', 'OCLC', 'ASIN', 'OSTI', 'MR', 'ZBL', 'SSRN', 'JSTOR',
    # 'BIBCODE', 'ISBN', 'USENETID', 'ISSN']

    for id_ in kinds_of_ids:
        citation_with_ids = citation_with_ids.withColumn(id_, lit(None))

    citation_with_ids = citation_with_ids.withColumn('actual_label', lit("rest"))

    for id_ in kinds_of_ids:
        def get_citation_val(x):
            for item in x:
                if item[0] == id_ and len(item) > 1:
                    return item[1]
            return None
        udf_get_citation_val = udf(get_citation_val)
        citation_with_ids = citation_with_ids.withColumn(id_, udf_get_citation_val("ID_list"))

    def get_label(doi, pmid, pmc, isbn, type):
        category = 'rest'
        if pmid or pmc:
            category = 'journal'
        if doi and not pmc and not pmid and not isbn:
            category = 'journal'
        if isbn and not pmc and not pmid and not doi:
            category = 'book'
        if isbn and doi:
            if type in ['cite journal', 'cite conference']:
                category = 'journal'
            elif type in ['cite book', 'cite encyclopedia']:
                category = 'book'
        return category

    udf_get_label = udf(get_label)
    citation_with_ids = citation_with_ids.withColumn('actual_label', udf_get_label('DOI', 'PMID', 'PMC', 'ISBN', 'type_of_citation'))
    citation_with_ids = citation_with_ids.filter(col('actual_label').isin(['book', 'journal']))

    for id_ in kinds_of_ids:
        citation_with_ids = citation_with_ids.drop(id_)
    citation_with_ids = citation_with_ids.drop("ID_list", "kinds_of_ids")

    def array_to_string(my_list):
        return '[' + ','.join([str(elem) for elem in my_list]) + ']'

    array_to_string_udf = udf(array_to_string)

    # NK Without this spark at gcloud fails to write the result on disk
    citation_with_ids = citation_with_ids.withColumn('neighboring_tags', array_to_string_udf("neighboring_tags"))
    citation_with_ids = citation_with_ids.withColumn('neighboring_words', array_to_string_udf("neighboring_words"))


    citation_with_ids = citation_with_ids.select(
      'type_of_citation', 'citations', 'id', 'ref_index', 'sections',
      'total_words', 'neighboring_tags', 'neighboring_words', 'actual_label')

    labels = ['doi', 'isbn', 'pmc', 'pmid', 'url', 'work', 'newspaper', 'website']
    for label in labels:
        def remove_bias(x):
            return re.sub(label + '\s{0,10}=\s{0,10}([^|]+)', label + ' = ', x)
        udf_remove_bias = udf(remove_bias)
        citation_with_ids = citation_with_ids.withColumn('citations', udf_remove_bias('citations'))

    citation_with_ids = citation_with_ids.dropDuplicates(['id', 'citations'])

    # def mismatched(x, y):
    #    return x != 'cite ' + y

    # udf_mismatched = udf(mismatched, BooleanType())
    # corrected = citation_with_ids.filter(udf_mismatched('type_of_citation', 'actual_label'))
    # print('The number of relabelled citations: {}'.format(corrected.count()))
    # 12 out of 190

    k = min(citation_with_ids.count(), 1000000)
    n = k/citation_with_ids.count()

    book_journal_features = citation_with_ids.sample(n)
    print('The number of unique citations_with_ids: {}'.format(book_journal_features.count()))

    book_journal_features = book_journal_features.limit()
    book_journal_features.write.mode('overwrite').parquet(file_out)

    # citation_with_ids.write.mode('overwrite').parquet(file_out)


