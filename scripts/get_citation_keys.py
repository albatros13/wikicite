from pyspark import SparkContext, SQLContext
from pyspark.sql import Row
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import udf, lit, col


def get_citation_keys(sql_context, file_in, file_out):
    print("Step 3: Creating citation dictionary...")

    generic_citations = sql_context.read.parquet(file_in)
    # NK changed: udf_get_keys = udf(lambda x: x.keys() if x.keys() is not None else [], ArrayType(StringType()))
    udf_get_keys = udf(lambda x: list(x.keys()) if x.keys() is not None else [], ArrayType(StringType()))

    # Get all the keys in the citation dict and remove that additional column from the additional DF
    generic_citations = generic_citations.withColumn('citation_keys', udf_get_keys(generic_citations.citation_dict))
    generic_citations.registerTempTable('generic_citations')

    citation_keys = sql_context.sql('select citation_keys as keys from generic_citations')
    key_iterator = citation_keys.rdd.toLocalIterator()

    # Get all the keys present in the Map for the citation template
    distinct_keys = set()
    for line in key_iterator:
        distinct_keys.update(line.keys)
    distinct_keys = list(distinct_keys)

    for key_ in distinct_keys:
        generic_citations = generic_citations.withColumn(key_, lit(None).cast(StringType()))
        
    def get_value(citation, key):
        return citation[key] if key in citation else ""

    def get_as_row(line):
        """
        Get each article's generic temolated citations with their id, title and type.
        :line: a row from the dataframe generated from get_data.py.
        """

        # NK was before
        # (city, title, issue, p_name, degree, format_, volume, authors,
        # date, pages, chron, chapter, url, p_place, id_list, encyclopedia,
        # series_number, access_date, series, edition, periodical, title_type) = get_value_from_citation(line.citation_dict)

        # Distinct keys now:
        # ['TitleType', 'Chapter', 'PublisherName', 'URL', 'Encyclopedia', 'Authors', 'ID_list', 'Edition',
        # 'Periodical', 'Issue', 'Date', 'Title', 'Pages', 'Format', 'Volume', 'PublicationPlace', 'Chron', 'Series']
        # Missing:
        # Degree, SeriesNumber, AccessDate, City,

        return Row(
            citations=line.citations,
            id=line.id,
            type_of_citation=line.type_of_citation,
            page_title=line.page_title,
            r_id=line.r_id,
            r_parentid=line.r_parentid,
            sections=line.sections,
            Degree=get_value(line.citation_dict, 'Degree'),
            City=get_value(line.citation_dict, 'City'),
            SeriesNumber=get_value(line.citation_dict, 'SeriesNumber'),
            AccessDate=get_value(line.citation_dict, 'AccessDate'),
            Chapter=get_value(line.citation_dict, 'Chapter'),
            PublisherName=get_value(line.citation_dict, 'PublisherName'),
            Format=get_value(line.citation_dict, 'Format'),
            Title=get_value(line.citation_dict, 'Title'),
            URL=get_value(line.citation_dict, 'URL'),
            Series=get_value(line.citation_dict, 'Series'),
            Authors=get_value(line.citation_dict, 'Authors'),
            ID_list=get_value(line.citation_dict, 'ID_list'),
            Encyclopedia=get_value(line.citation_dict, 'Encyclopedia'),
            Periodical=get_value(line.citation_dict, 'Periodical'),
            PublicationPlace=get_value(line.citation_dict, 'PublicationPlace'),
            Date=get_value(line.citation_dict, 'Date'),
            Edition=get_value(line.citation_dict, 'Edition'),
            Pages=get_value(line.citation_dict, 'Pages'),
            Chron=get_value(line.citation_dict, 'Chron'),
            Issue=get_value(line.citation_dict, 'Issue'),
            Volume=get_value(line.citation_dict, 'Volume'),
            TitleType=get_value(line.citation_dict, 'TitleType')
        )

    generic_citations = sql_context.createDataFrame(generic_citations.rdd.map(get_as_row), samplingRatio=0.2)
    generic_citations.write.mode('overwrite').parquet(file_out)


def get_citation_ids(sql_context, file_in, file_out):
    generic_citations = sql_context.read.parquet(file_in)
    # Code to get CSV file for some particular column which only have ID List
    id_list_exists = generic_citations.where(col('ID_list').isNotNull())
    id_list_exists = id_list_exists.select('id', 'page_title', 'citations', 'ID_list', 'Authors',
                                           'Title', 'type_of_citation', 'PublisherName', 'sections')
    # NK sanity_check expects the following columns:
    # 'id', 'page_title', 'citation', 'id_list', 'authors',
    # 'citation_title', 'citation_type', 'publisher_name', 'sections'
    id_list_exists = id_list_exists.withColumnRenamed('citations', 'citation')\
        .withColumnRenamed('ID_list', 'id_list') \
        .withColumnRenamed('Authors', 'authors')\
        .withColumnRenamed('Title', 'citation_title')\
        .withColumnRenamed('type_of_citation', 'citation_type')\
        .withColumnRenamed('PublisherName', 'publisher_name') \

    id_list_exists.write.mode('overwrite').parquet(file_out)
