import os
import unittest
import pandas as pd
from test_helpers import get_parsed_citations
import mwparserfromhell


class TestCorrectNumberOfReferences(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        TEST_DATA_PATH = 'data/random_pages.csv/'
        files = []
        for name in os.listdir(TEST_DATA_PATH):
            if name.startswith("part"):
                files.append(TEST_DATA_PATH + name)
        random_pages_dfs = [pd.read_csv(f, header=None, sep=',') for f in files if os.path.getsize(f) > 0]
        # Considering just the title and content of the page
        cls.random_pages = pd.concat(random_pages_dfs, ignore_index=True)[[1, 2]]

    def test_check_refs_page_one(self):
        title, content = self.random_pages.iloc[0]
        # print(content)
        parsed_cites = get_parsed_citations(content)
        manual_counting = 3
        self.assertEqual(len(parsed_cites), manual_counting)
        # for p in parsed_cites:
        #     print(p)
        self.assertTrue(any([t == 'harvnb' for _,t in parsed_cites]))

    def test_check_refs_page_two(self):
        title, content = self.random_pages.iloc[1]
        # print(content)
        parsed_cites = get_parsed_citations(content)
        manual_counting = 0
        self.assertEqual(len(parsed_cites), manual_counting)
        for p in parsed_cites:
            print(p)
        self.assertTrue('cite DNB' in content)

    def test_check_refs_page_three(self):
        title, content = self.random_pages.iloc[2]
        # print(content)
        parsed_cites = get_parsed_citations(content)
        manual_counting = 0
        self.assertEqual(len(parsed_cites), manual_counting)
        self.assertFalse('cite' in content)

    def test_check_refs_page_four(self):
        title, content = self.random_pages.iloc[3]
        parsed_cites = get_parsed_citations(content)
        # for c in parsed_cites:
        #     print(c)
        manual_counting = 14
        self.assertEqual(len(parsed_cites), manual_counting)
        self.assertTrue(all([t == 'cite web' for _, t in parsed_cites]))

    def test_check_refs_page_five(self):
        # This is about a city in Iran - a test case for geolocation
        title, content = self.random_pages.iloc[4]
        parsed_cites = get_parsed_citations(content)
        manual_counting = 1
        # Added a test citation from a new format like geonet3 - naturally
        self.assertEqual(len(parsed_cites), manual_counting)
        # One of the citations is a harvnb
        self.assertTrue(
            any([t == 'geonet3' for _,t in parsed_cites]) and 'IranCensus2006' in content
        )

    def test_wikiciteparser(self):
        mwtext = """
            ===Articles===
             * {{Citation | last1=Lambek | first1=Joachim | author1-link=Joachim Lambek | last2=Moser | first2=L. | title=Inverse and Complementary Sequences of Natural Numbers| doi=10.2307/2308078 | mr=0062777  | journal=[[American Mathematical Monthly|The American Mathematical Monthly]] | issn=0002-9890 | volume=61 | issue=7 | pages=454–458 | year=1954 | jstor=2308078 | publisher=The American Mathematical Monthly, Vol. 61, No. 7}}
             * {{Citation | last1=Lambek | first1=J. | author1-link=Joachim Lambek | title=The Mathematics of Sentence Structure | year=1958 | journal=[[American Mathematical Monthly|The American Mathematical Monthly]] | issn=0002-9890 | volume=65 | pages=154–170 | doi=10.2307/2310058 | issue=3 | publisher=The American Mathematical Monthly, Vol. 65, No. 3 | jstor=1480361}}
             *{{Citation | last1=Lambek | first1=Joachim | author1-link=Joachim Lambek | title=Bicommutators of nice injectives | doi=10.1016/0021-8693(72)90034-8 | mr=0301052  | year=1972 | journal=Journal of Algebra | issn=0021-8693 | volume=21 | pages=60–73}}
             *{{Citation | last1=Lambek | first1=Joachim | author1-link=Joachim Lambek | title=Localization and completion | doi=10.1016/0022-4049(72)90011-4 | mr=0320047  | year=1972 | journal=Journal of Pure and Applied Algebra | issn=0022-4049 | volume=2 | pages=343–370 | issue=4}}
             *{{Citation | last1=Lambek | first1=Joachim | author1-link=Joachim Lambek | title=A mathematician looks at Latin conjugation | mr=589163  | year=1979 | journal=Theoretical Linguistics | issn=0301-4428 | volume=6 | issue=2 | pages=221–234 | doi=10.1515/thli.1979.6.1-3.221}}
            """

        wikicode = mwparserfromhell.parse(mwtext)
        tpls = wikicode.filter_templates()
        self.assertEqual(len(tpls), 5)
        for tpl in tpls:
            self.assertEqual(tpl.name.strip(), 'Citation')
            # parsed = parse_citation_template(tpl)
            # print(parsed)


if __name__ == '__main__':
    unittest.main()
