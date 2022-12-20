# -*- coding: utf-8 -*-
"""
For each citation check if they conform to a template as given in scripts/const.py
and then extract features for all of them based on their content.
"""

import re
import mwparserfromhell
from nltk import pos_tag
from pyspark.sql.functions import explode
from pyspark.sql.types import *


def extract_nlp_features(sql_context, file_in, file_out):
    print("Step 5: Extracting NLP features...")

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
