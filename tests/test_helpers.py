import unittest
import re
# Using the forked libraries
from wikiciteparser.parser import parse_citation_template
import mwparserfromhell


def get_parsed_citations(content):
    parsed_cites = []

    # Go through each of the templates
    wikicode = mwparserfromhell.parse(content)
    templates = wikicode.filter_templates()
    for tpl in templates:
        citation = parse_citation_template(tpl)
        if citation:
            type_of_citation = tpl.split('|')[0].lower()[2:]
            parsed_cites.append((citation, type_of_citation))
    return parsed_cites


def get_citations(page_content):
    wikicode = mwparserfromhell.parse(page_content)
    templates = wikicode.filter_templates()
    citations = []
    for tpl in templates:
        if tpl.startswith('{{') or tpl.startswith("'{{"):
            _tpl = repr(tpl)
            citations.append(_tpl)
    return citations


def check_if_balanced(my_string):
    """
    Check if particular citation has balanced brackets.
    :param: citation to be taken in consideration
    """
    my_string = re.sub('\w|\s|[^{}]', '', my_string)
    brackets = ['()', '{}', '[]']
    while any(x in my_string for x in brackets):
        for br in brackets:
            my_string = my_string.replace(br, '')
    return not my_string


class TestHelper(unittest.TestCase):

    def test_check_if_balanced(self):
        self.assertEqual(check_if_balanced('{{}}'), True)

    def test_check_if_not_balanced(self):
        self.assertEqual(check_if_balanced('{{{}}}}'), False)

    def test_check_if_balanced_error(self):
        with self.assertRaises(TypeError):
            check_if_balanced(88)


if __name__ == '__main__':
    unittest.main()
