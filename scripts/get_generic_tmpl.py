import mwparserfromhell
from const import CITATION_TEMPLATES
from wikiciteparser.parser import parse_citation_template
from pyspark.sql.functions import expr, udf, col, lit
import re
from pyspark.sql.types import *


def get_generic_tmpl(sql_context, file_in, file_out):
    print("Step 4: Converting citations to generic template...")

    sql_context.setConf('spark.sql.parquet.compression.codec', 'snappy')

    citations = sql_context.read.parquet(file_in)
    citations = citations.withColumn('type_of_citation', expr('substring(type_of_citation, 2, length(type_of_citation))'))

    print("Before matching with templates:", citations.count(), len(citations.columns))
    # NK what is the number of citations before filtering?
    citations = citations.filter(citations['type_of_citation'].isin(CITATION_TEMPLATES))
    print("After matching with templates:", citations.count(), len(citations.columns))

    def check_if_balanced(my_string):
        """
        Check if particular citation has balanced brackets.
        :param: citation to be taken in consideration
        """
        my_string = re.sub('\w|\s|[^{}]','', my_string)
        brackets = ['()', '{}', '[]']
        while any(x in my_string for x in brackets):
            for br in brackets:
                my_string = my_string.replace(br, '')
        return not my_string

    # NK Note: in some files empty authors are []
    def preprocess_authors(authors):
        return ", ".join([str(a) for a in authors])

    def get_generic_template(citation):
        """
            Get generic template of a citation using the wikiciteparser library.
            :param: citation - according to a particular format as described in const.py
        """
        not_parsable = {'Title': 'Citation generic template not possible'}
        if not check_if_balanced(citation):
            citation = citation + '}}'
        try:
            # Convert the str into a mwparser object
            wikicode = mwparserfromhell.parse(citation)
            template = wikicode.filter_templates()[0]
            parsed_result = parse_citation_template(template, 'en')
        except Exception as e:
            print("Failed to parse citation template: ", e)
            return not_parsable

        # NK This is a fix for potentially different field types: array vs string
        if "Authors" in parsed_result:
            parsed_result["Authors"] = preprocess_authors(parsed_result["Authors"])

        return parsed_result

    def get_value(citation, key):
        if key in citation:
            if citation[key] is not None:
                return citation[key]
        return ""

    def get_as_row(line):
        """
            Get each article's generic templated citations with their id, title and type.
            :param line: a row from the dataframe generated from get_data.py.
        """
        citation_dict = get_generic_template(line.citations)
        return Row(
            citations=line.citations,
            id=line.id,
            type_of_citation=line.type_of_citation,
            page_title=line.title,
            r_id=line.r_id,
            r_parentid=line.r_parentid,
            sections=line.sections,
            Degree=get_value(citation_dict, 'Degree'),
            City=get_value(citation_dict, 'City'),
            SeriesNumber=get_value(citation_dict, 'SeriesNumber'),
            AccessDate=get_value(citation_dict, 'AccessDate'),
            Chapter=get_value(citation_dict, 'Chapter'),
            PublisherName=get_value(citation_dict, 'PublisherName'),
            Format=get_value(citation_dict, 'Format'),
            Title=get_value(citation_dict, 'Title'),
            URL=get_value(citation_dict, 'URL'),
            Series=get_value(citation_dict, 'Series'),
            Authors=get_value(citation_dict, 'Authors'),
            ID_list=get_value(citation_dict, 'ID_list'),
            Encyclopedia=get_value(citation_dict, 'Encyclopedia'),
            Periodical=get_value(citation_dict, 'Periodical'),
            PublicationPlace=get_value(citation_dict, 'PublicationPlace'),
            Date=get_value(citation_dict, 'Date'),
            Edition=get_value(citation_dict, 'Edition'),
            Pages=get_value(citation_dict, 'Pages'),
            Chron=get_value(citation_dict, 'Chron'),
            Issue=get_value(citation_dict, 'Issue'),
            Volume=get_value(citation_dict, 'Volume'),
            TitleType=get_value(citation_dict, 'TitleType')
        )

    generic_citations = sql_context.createDataFrame(citations.rdd.map(get_as_row))
    generic_citations.write.mode('overwrite').parquet(file_out)


def only_with_ids(sql_context, file_in, file_out):
    print("Step 4a: Saving citations in csv...")
    generic_citations = sql_context.read.parquet(file_in)

    id_list_exists = generic_citations.where(col('ID_list').isNotNull())
    id_list_exists = id_list_exists.select('id', 'page_title', 'citations', 'ID_list', 'Authors',
                                           'Title', 'type_of_citation', 'PublisherName', 'sections')

    id_list_exists = id_list_exists.withColumnRenamed('citations', 'citation') \
        .withColumnRenamed('ID_list', 'id_list') \
        .withColumnRenamed('Authors', 'authors') \
        .withColumnRenamed('Title', 'citation_title') \
        .withColumnRenamed('type_of_citation', 'citation_type') \
        .withColumnRenamed('PublisherName', 'publisher_name')

    id_list_exists.write.mode('overwrite').parquet(file_out)

