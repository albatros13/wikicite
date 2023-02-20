from pyspark.sql.functions import explode, col, split, trim, lower, regexp_replace, expr, udf, lit
from pyspark.sql.types import Row, StructType, StructField, ArrayType, StringType, BooleanType
import mwparserfromhell
from pyspark import SparkContext, SQLContext
import os
import sys
import findspark
import re
from nltk import pos_tag
from wikiciteparser.parser import parse_citation_template
import tldextract


os.environ["PYSPARK_SUBMIT_ARGS"] = " --packages com.databricks:spark-xml_2.12:0.15.0 pyspark-shell"
findspark.init()

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sc = SparkContext.getOrCreate()
sql_context = SQLContext(sc)

# import nltk
# nltk.download('popular')

CITATION_TEMPLATES = set([
    'citation',
    'cite arxiv',
    'cite av media',
    'cite av media notes',
    'cite book',
    'cite conference',
    'cite dvd notes',
    'cite encyclopedia',
    'cite episode',
    'cite interview',
    'cite journal',
    'cite mailing list',
    'cite map',
    'cite news',
    'cite newsgroup',
    'cite podcast',
    'cite press release',
    'cite report',
    'cite serial',
    'cite sign',
    'cite speech',
    'cite techreport',
    'cite thesis',
    'cite web'
])


def get_data(file_in, file_out, file_out2, limit=None):
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


def extract_nlp_features(file_in, file_out):
    print("Step 3: Extracting NLP features...")

    TOTAL_NEIGHBORING_WORDS = 40
    PUNCTUATION_TO_BE_REMOVED = '"\[\\]^`|~'
    PUNC_REGEX = re.compile(r'[{}]+'.format(re.escape(PUNCTUATION_TO_BE_REMOVED)))

    citations_content = sql_context.read.parquet(file_in).repartition(400)

    def get_features(page_content):
        wikicode = mwparserfromhell.parse(page_content)
        templates = wikicode.filter_templates()

        # Get all words for a page associated with a title
        all_words = wikicode.get_all_tokens_feature()
        all_words = [repr(w) for w in all_words]
        # NK
        # all_words = [w[2:len(w) - 1] for w in all_words]
        all_words = [w[1:len(w) - 1] for w in all_words]

        all_words = [
            PUNC_REGEX.sub(' ', str(word)).replace(',', ' ').replace('//', '')
            if not word.startswith('{{') else word for word in all_words
        ]

        # and then remove the strings which are not None and have length greater than one
        all_words = [word for word in all_words if len(word) > 1 or word in list(':?@-.!')]
        # Get part of speech tags for the neighboring words for the citation
        words_plus_tags = [(w,t) for w, t in pos_tag(all_words)]
        # Set tag to be WIKICODE if it is a citation or a wikicode
        words_plus_tags = [(w, 'WIKICODE') if w.startswith('{{') else (w,t) for w, t in words_plus_tags]
        # return words_plus_tags
        total_words_in_page = len(words_plus_tags)
        results = []
        for tpl in templates:
            tpl = repr(tpl)
            # NK
            tpl = tpl[1:len(tpl) - 1]
            if tpl.startswith('{{'):
                if tpl in all_words:
                    ref_index = all_words.index(tpl)
                    if ref_index < TOTAL_NEIGHBORING_WORDS:
                        neighboring_before_words = words_plus_tags[:ref_index]
                    else:
                        neighboring_before_words = words_plus_tags[ref_index - TOTAL_NEIGHBORING_WORDS:ref_index]
                    results.append((tpl, ref_index, total_words_in_page, neighboring_before_words))
        return results

    def get_as_row(line):
        citations_features = get_features(line.content)
        return Row(citations_features=citations_features, page_title=line.page_title, id=line.id)

    citations_content = sql_context.createDataFrame(citations_content.rdd.map(get_as_row))
    citations_content = citations_content.withColumn('citations_features', explode('citations_features'))
    citations_content.write.mode('overwrite').parquet(file_out)


def get_generic_tmpl(file_in, file_out, lang='en'):
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

    def list_to_str(items):
        return ", ".join([str(a) for a in items])

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
            parsed_result = parse_citation_template(template, lang)
        except Exception as e:
            print("Failed to parse citation template: ", e)
            return not_parsable

        # NK This is a fix for potentially different field types: array vs string
        if "Authors" in parsed_result:
            parsed_result["Authors"] = list_to_str(parsed_result["Authors"])
        if "ID_list" in parsed_result:
            parsed_result["ID_list"] = str(parsed_result["ID_list"])
        if "PublisherName" in parsed_result:
            parsed_result["PublisherName"] = parsed_result["PublisherName"].replace("[[",'').replace("]]",'')

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


def get_dataset_features(file_in1, file_in2, file_out):
    print("Step 5: Getting dataset features...")

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


# Select entries with won-empty ID_list
def filter_with_ids(file_in, file_out):
    print("Step 6a: Selecting citations with identifiers...")

    sql_context.setConf('spark.sql.parquet.compression.codec', 'snappy')
    citations_features = sql_context.read.parquet(file_in)

    print("Length full:", citations_features.count())
    print('The columns in the citations features are: {}'.format(citations_features.columns))

    citation_with_ids = citations_features.where(col("ID_list") != '')
    citation_with_ids = citation_with_ids.dropDuplicates()
    print("Length filtered:", citation_with_ids.count())
    citation_with_ids.write.mode('overwrite').parquet(file_out)


def get_book_journal_features(file_in, file_out):
    print("Step 6b: Getting book and journal citations...")

    sql_context.setConf('spark.sql.parquet.compression.codec', 'snappy')
    citation_with_ids = sql_context.read.parquet(file_in)
    citation_with_ids = citation_with_ids.limit(1000000)

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
    print('The total number of citations_with_ids: {}'.format(citation_with_ids.count()))

    for id_ in kinds_of_ids:
        citation_with_ids = citation_with_ids.drop(id_)
    citation_with_ids = citation_with_ids.drop("ID_list", 'kinds_of_ids')

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
    print('The number of unique citations_with_ids: {}'.format(citation_with_ids.count()))
    citation_with_ids.write.mode('overwrite').parquet(file_out)


def get_newspaper_citations(file_in, file_out):
    print("Step 7: Getting newspaper citations...")

    sql_context.setConf('spark.sql.parquet.compression.codec', 'snappy')
    citations_separated = sql_context.read.parquet(file_in)

    citations_separated = citations_separated.where(col("URL").isNotNull())

    def get_top_domain(citation_url):
        url_ext = tldextract.extract(citation_url)
        return url_ext.domain

    top_domain_udf = udf(get_top_domain)
    citations_separated = citations_separated.withColumn('tld', top_domain_udf('URL'))

    # NK this is the original list of top news agency domains - I use a longer list
    # newspapers = {'nytimes', 'bbc', 'washingtonpost', 'cnn', 'theguardian', 'huffingtonpost', 'indiatimes',
    #               'independent', 'dailymail', 'telegraph', 'timesonline', 'reuters'}

    # NK distinct set of news agency domains from 'cite news' dataset
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


def get_selected_features(file_in1, file_in2, file_out):
    print("Step 8: Getting selected features...")

    sql_context.setConf('spark.sql.parquet.compression.codec', 'snappy')
    features = sql_context.read.parquet(file_in1)
    features = features.withColumnRenamed('page_title', 'page_title_')

    features = features.select(
        col('citations_features._1').alias('retrieved_citation'),
        col('citations_features._2').alias('ref_index'),
        col('citations_features._3').alias('total_words'),
        col('citations_features._4._1').alias('neighboring_words'),
        col('citations_features._4._2').alias('neighboring_tags'),
    )

    selected_features = sql_context.read.parquet(file_in2)

    def array_to_string(my_list):
       return '[' + ','.join([str(elem) for elem in my_list]) + ']'

    array_to_string_udf = udf(array_to_string,StringType())

    results = features.join(selected_features, features['retrieved_citation'] == selected_features['citations'])
    results = results.withColumn('neighboring_words', array_to_string_udf(results["neighboring_words"]))
    results = results.withColumn('neighboring_tags', array_to_string_udf(results["neighboring_tags"]))

    results = results.drop('retrieved_citation')
    results.write.mode('overwrite').parquet(file_out)


# Files
PROJECT_HOME = "gs://wikicite-1/"
# PROJECT_HOME = 'c:///users/natal/PycharmProjects/cite-classifications-wiki/'
ext = "en_"
INPUT_DATA = PROJECT_HOME + 'data/dumps/enwiki-20221201-pages-articles-multistream1.xml-p1p41242.bz2'
# ext = "xh_"
# INPUT_DATA = PROJECT_HOME + 'data/dumps/xhwiki-20221001-pages-articles-multistream.xml.bz2'

CITATIONS = PROJECT_HOME + 'data/content/{}citations.parquet'.format(ext)
CITATIONS_CONTENT = PROJECT_HOME + 'data/content/{}citations_content.parquet'.format(ext)
BASE_FEATURES = PROJECT_HOME + 'data/features/{}base_features_complete.parquet'.format(ext)
CITATIONS_SEPARATED = PROJECT_HOME + 'data/content/{}citations_separated.parquet'.format(ext)
CITATIONS_FEATURES = PROJECT_HOME + 'data/features/{}citations_features.parquet'.format(ext)
CITATIONS_FEATURES_IDS = PROJECT_HOME + 'data/features/{}citations_features_ids.parquet'.format(ext)
BOOK_JOURNAL_CITATIONS = PROJECT_HOME + 'data/features/{}book_journal_citations.parquet'.format(ext)
NEWSPAPER_CITATIONS = PROJECT_HOME + 'data/features/{}newspaper_citations.parquet'.format(ext)
NEWSPAPER_FEATURES = PROJECT_HOME + 'data/features/{}newspaper_citation_features.parquet'.format(ext)


# Data extraction steps
get_data(INPUT_DATA, CITATIONS, CITATIONS_CONTENT)
extract_nlp_features(CITATIONS_CONTENT, BASE_FEATURES)
get_generic_tmpl(CITATIONS, CITATIONS_SEPARATED)
get_dataset_features(BASE_FEATURES, CITATIONS_SEPARATED, CITATIONS_FEATURES)

# Labelled datasets for classifier training
filter_with_ids(CITATIONS_FEATURES, CITATIONS_FEATURES_IDS)
get_book_journal_features(CITATIONS_FEATURES_IDS, BOOK_JOURNAL_CITATIONS)
get_newspaper_citations(CITATIONS_SEPARATED, NEWSPAPER_CITATIONS)
get_selected_features(BASE_FEATURES, NEWSPAPER_CITATIONS, NEWSPAPER_FEATURES)