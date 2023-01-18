from pyspark.sql.functions import explode, col, split, trim, lower, regexp_replace
from pyspark.sql.types import *
import mwparserfromhell


def get_data(sql_context, file_in, file_out, file_out2, limit=None):
    print("Step 1: Getting citations from XML dump...")

    sql_context.setConf('spark.sql.parquet.compression.codec', 'snappy')

    wiki = sql_context.read.format('com.databricks.spark.xml').options(rowTag='page').load(file_in)
    pages_all = wiki.where('ns = 0').where('redirect is null')

    # Get only ID, title, revision text's value which we are interested in
    pages = pages_all['id', 'title', 'revision.text', 'revision.id', 'revision.parentid']
    pages = pages.toDF('id', 'title', 'content', 'r_id', 'r_parentid')

    def get_citations(page_content):
        """
            Get the <ref></ref> tag citations and citations which are standalone in a format, for e.g. "* {{"
            :param: page_content: <text></text> part of the wikicode xml.
        """
        wikicode = mwparserfromhell.parse(page_content)
        templates = wikicode.filter_templates()
        section_features = wikicode.get_node_sections_feature()

        citations = []
        sections = []
        for tpl in templates:
            # c_exists = regex.findall(CITATION_REGEX, repr(tpl))
            # NK String representation changed in Python 3
            if tpl.startswith('{{') or tpl.startswith("'{{"):
                _tpl = repr(tpl)
                citations.append(_tpl)
                sections.append(', '.join(section_features[_tpl]))
        return list(zip(citations, sections))

    def get_as_row(line):
        """
         Get each article's citations with their id and title.
         :param line: the wikicode for the article
        """
        citations = get_citations(line.content)
        return Row(citations=citations, id=line.id, title=line.title, r_id=line.r_id, r_parentid=line.r_parentid)

    schema = StructType([
        StructField("citations", ArrayType(
            StructType([
                StructField("_1", StringType(), True),
                StructField("_2", StringType(), True)
            ])
        )),
        StructField("id", StringType(), True),
        StructField("title", StringType(), True),
        StructField("r_id", StringType(), True),
        StructField("r_parentid", StringType(), True)
    ])

    cite_df = sql_context.createDataFrame(pages.rdd.map(get_as_row), schema=schema)
    cite_df = cite_df.withColumn('citations', explode('citations'))
    # cite_df = cite_df.withColumn('tmp', zip_('citations', 'sections')).withColumn('tmp', explode('tmp'))
    cite_df = cite_df.select('id', 'r_id', 'r_parentid', 'title', col('citations._1').alias('citations'), col('citations._2').alias('sections'))
    split_col = split(cite_df['citations'], '\|')
    cite_df = cite_df.withColumn('type_of_citation', lower(trim(split_col.getItem(0))))
    cite_df = cite_df.withColumn('type_of_citation', regexp_replace('type_of_citation', '\{\{', ''))

    if limit:
        cite_df = cite_df.limit(limit)
    cite_df.write.mode('overwrite').parquet(file_out)

    #################################################

    print("Step 2: Getting content from XML dump...")

    # Get only ID, title, revision text's value which we are interested in
    pages = pages_all['id', 'title', 'revision.text']
    pages = pages.toDF('id', 'page_title', 'content')

    if limit:
        pages = pages.limit(limit)
    pages.write.mode('overwrite').parquet(file_out2)