import mwparserfromhell
from pyspark.sql import Row
from const import CITATION_TEMPLATES
from wikiciteparser.parser import parse_citation_template
from pyspark.sql.functions import expr, regexp_replace
import re
from pyspark.sql.types import *


def get_generic_tmpl(sql_context, file_in, file_out):
    print("Step 2: Converting citations to generic template...")

    sql_context.setConf('spark.sql.parquet.compression.codec', 'snappy')

    citations = sql_context.read.parquet(file_in)
    # citations = citations.withColumn('type_of_citation', expr('substring(type_of_citation, 3, length(type_of_citation))'))
    citations = citations.withColumn('type_of_citation', expr('substring(type_of_citation, 2, length(type_of_citation))'))
    citations = citations.filter(citations['type_of_citation'].isin(CITATION_TEMPLATES))

    # NK TODO Lua fails because of 'vauthors' instead of 'authors'

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

    def get_generic_template(citation):
        """
            Get generic template of a citation using the wikiciteparser library.
            :param: citation - according to a particular format as described in const.py
        """
        not_parsable = {'Title': 'Citation generic template not possible'}
        if not check_if_balanced(citation):
            citation = citation + '}}'
        # Convert the str into a mwparser object
        lang = 'en'
        try:
            wikicode = mwparserfromhell.parse(citation)
            template = wikicode.filter_templates()[0]
            parsed_result = parse_citation_template(template, lang)
        # except IndexError:
        # ValueError
        except Exception:
            return not_parsable

        # NK fix issues with saving parquet file?
        # for key in parsed_result:
        #   parsed_result[key] = str(parsed_result[key])
        # if "URL" in parsed_result:
        #     del parsed_result["URL"]

        # In case the mwparser is not able to parse the citation template
        return parsed_result

    def get_as_row(line):
        """
            Get each article's generic temolated citations with their id, title and type.
            :param line: a row from the dataframe generated from get_data.py.
        """
        row = Row(
            citation_dict=get_generic_template(line.citations),
            # citation_dict={'Title': 'Citation generic template not possible'}, # NK this fixes the issue!
            id=line.id,
            page_title=line.title, type_of_citation=line.type_of_citation,
            r_id=line.r_id, r_parentid=line.r_parentid,
            citations=line.citations, sections=line.sections
        )
        return row

    generic_citations =sql_context.createDataFrame(citations.rdd.map(get_as_row))
    generic_citations.write.mode('overwrite').parquet(file_out)
