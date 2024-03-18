# -*- encoding: utf-8 -*-
from __future__ import unicode_literals

import unittest
import mwparserfromhell
from wikiciteparser.parser import *

# from translator import translate_and_parse_citation_template


class ParsingTests(unittest.TestCase):
    @unittest.skip("Not relevant for multi-language parsing")
    def test_multiple_authors(self):
        p = parse_citation_dict({"doi": "10.1111/j.1365-2486.2008.01559.x",
                                 "title": "Climate change, plant migration, and range collapse in a global biodiversity hotspot: the ''Banksia'' (Proteaceae) of Western Australia",
                                 "issue": "6",
                                 "journal": "Global Change Biology",
                                 "year": "2008",
                                 "volume": "14",
                                 "last4": "Dunn",
                                 "last1": "Fitzpatrick",
                                 "last3": "Sanders",
                                 "last2": "Gove", "first1":
                                 "Matthew C.",
                                 "first2": "Aaron D.",
                                 "first3": "Nathan J.",
                                 "first4": "Robert R.",
                                 "pages": "1\u201316"
                                 },
                                template_name='cite journal')
        self.assertEqual(p['Authors'], [{'last': 'Fitzpatrick',
                                         'first': 'Matthew C.'
                                         },
                                        {'last': 'Gove',
                                         'first': 'Aaron D.'},
                                        {'last': 'Sanders',
                                         'first': 'Nathan J.'},
                                        {'last': 'Dunn',
                                         'first': 'Robert R.'
                                         }
                                        ])

    @unittest.skip("Not relevant for multi-language parsing")
    def test_vauthors(self):
        p = parse_citation_dict({"doi": "10.1016/s1097-2765(00)80111-2",
                                 "title": "SAP30, a component of the mSin3 corepressor complex involved in N-CoR-mediated repression by specific transcription factors",
                                 "journal": "Mol. Cell",
                                 "volume": "2",
                                 "date": "July 1998",
                                 "pmid": "9702189",
                                 "issue": "1",
                                 "pages": "33\u201342",
                                 "vauthors": "Laherty CD, Billin AN, Lavinsky RM, Yochum GS, Bush AC, Sun JM, Mullen TM, Davie JR, Rose DW, Glass CK, Rosenfeld MG, Ayer DE, Eisenman RN"
                                 },
                                template_name='cite journal')
        self.assertEqual(p['Authors'], [{'last': 'Laherty',
                                         'first': 'CD'
                                         },
                                        {'last': 'Billin',
                                         'first': 'AN'
                                         },
                                        {'last': 'Lavinsky',
                                         'first': 'RM'
                                         },
                                        {'last': 'Yochum',
                                         'first': 'GS'
                                         },
                                        {'last': 'Bush',
                                         'first': 'AC'
                                         },
                                        {'last': 'Sun',
                                         'first': 'JM'
                                         },
                                        {'last': 'Mullen',
                                         'first': 'TM'
                                         },
                                        {'last': 'Davie',
                                         'first': 'JR'
                                         },
                                        {'last': 'Rose',
                                         'first': 'DW'
                                         },
                                        {'last': 'Glass',
                                         'first': 'CK'
                                         },
                                        {'last': 'Rosenfeld',
                                         'first': 'MG'
                                         },
                                        {'last': 'Ayer',
                                         'first': 'DE'
                                         },
                                        {'last': 'Eisenman',
                                         'first': 'RN'
                                         }
                                        ])

    @unittest.skip("Not relevant for multi-language parsing")
    def test_remove_links(self):
        p = parse_citation_dict({"title": "Mobile, Alabama",
                                 "url": "http://archive.org/stream/ballouspictorial1112ball#page/408/mode/2up",
                                 "journal": "[[Ballou's Pictorial Drawing-Room Companion]]",
                                 "volume": "12",
                                 "location": "Boston",
                                 "date": "June 27, 1857"
                                 },
                                template_name='cite journal')
        self.assertEqual(p['Periodical'],
                         "Ballou's Pictorial Drawing-Room Companion")

    @unittest.skip("Not relevant for multi-language parsing")
    def test_authorlink(self):
        p = parse_citation_dict({"publisher": "[[World Bank]]",
                                 "isbn": "978-0821369418",
                                 "title": "Performance Accountability and Combating Corruption",
                                 "url": "http://siteresources.worldbank.org/INTWBIGOVANTCOR/Resources/DisruptingCorruption.pdf",
                                 "page": "309",
                                 "last1": "Shah",
                                 "location": "[[Washington, D.C.]], [[United States|U.S.]]",
                                 "year": "2007",
                                 "first1": "Anwar",
                                 "authorlink1": "Anwar Shah",
                                 "oclc": "77116846"
                                 },
                                template_name='citation')
        self.assertEqual(p['Authors'], [{'link': 'Anwar Shah',
                                         'last': 'Shah',
                                         'first': 'Anwar'
                                         }
                                        ])

    @unittest.skip("Not relevant for multi-language parsing")
    def test_unicode(self):
        p = parse_citation_dict({"title": "\u0414\u043e\u0440\u043e\u0433\u0438 \u0446\u0430\u0440\u0435\u0439 (Roads of Emperors)",
                                 "url": "http://magazines.russ.ru/ural/2004/10/mar11.html",
                                 "journal": "\u0423\u0440\u0430\u043b",
                                 "author": "Margovenko, A",
                                 "volume": "10",
                                 "year": "2004"
                                 },
                                template_name='cite journal')
        self.assertEqual(p['Title'],
                         '\u0414\u043e\u0440\u043e\u0433\u0438 \u0446\u0430\u0440\u0435\u0439 (Roads of Emperors)')

    def test_mwtext(self):
        # taken from https://en.wikipedia.org/wiki/Joachim_Lambek
        mwtext = """
        ===Articles===
        * {{Citation | last1=Lambek | first1=Joachim | author1-link=Joachim Lambek | last2=Moser | first2=L. | title=Inverse and Complementary Sequences of Natural Numbers| doi=10.2307/2308078 | mr=0062777  | journal=[[American Mathematical Monthly|The American Mathematical Monthly]] | issn=0002-9890 | volume=61 | issue=7 | pages=454–458 | year=1954 | jstor=2308078 | publisher=The American Mathematical Monthly, Vol. 61, No. 7}}
        * {{Citation | last1=Lambek | first1=J. | author1-link=Joachim Lambek | title=The Mathematics of Sentence Structure | year=1958 | journal=[[American Mathematical Monthly|The American Mathematical Monthly]] | issn=0002-9890 | volume=65 | pages=154–170 | doi=10.2307/2310058 | issue=3 | publisher=The American Mathematical Monthly, Vol. 65, No. 3 | jstor=1480361}}
        *{{Citation | last1=Lambek | first1=Joachim | author1-link=Joachim Lambek | title=Bicommutators of nice injectives | doi=10.1016/0021-8693(72)90034-8 | mr=0301052  | year=1972 | journal=Journal of Algebra | issn=0021-8693 | volume=21 | pages=60–73}}
        *{{Citation | last1=Lambek | first1=Joachim | author1-link=Joachim Lambek | title=Localization and completion | doi=10.1016/0022-4049(72)90011-4 | mr=0320047  | year=1972 | journal=Journal of Pure and Applied Algebra | issn=0022-4049 | volume=2 | pages=343–370 | issue=4}}
        *{{Citation | last1=Lambek | first1=Joachim | author1-link=Joachim Lambek | title=A mathematician looks at Latin conjugation | mr=589163  | year=1979 | journal=Theoretical Linguistics | issn=0301-4428 | volume=6 | issue=2 | pages=221–234 | doi=10.1515/thli.1979.6.1-3.221}}
        {{cite journal | vauthors = Caboche M, Bachellerie JP | title = RNA methylation and control of eukaryotic RNA biosynthesis. Effects of cycloleucine, a specific inhibitor of methylation, on ribosomal RNA maturation | journal = European Journal of Biochemistry | volume = 74 | issue = 1 | pages = 19–29 | date = March 1977 | pmid = 856572 | doi = 10.1111/j.1432-1033.1977.tb11362.x | doi-access = free }}
        """
        wikicode = mwparserfromhell.parse(mwtext)
        for tpl in wikicode.filter_templates():
            parsed = parse_citation_template(tpl, 'en')
            print(parsed)
            # All templates in this example are citation templates
            self.assertIsInstance(parsed, dict)

    def test_translate_it(self):
        mwtext = """
                {{cita libro | autore = Barth, F. | titolo = Ethnic groups and boundaries: The social organization of culture differences | url = https://archive.org/details/ethnicgroupsboun0000unse | lingua = en | editore = Little Brown & Co. | città = Boston | anno = 1969}}
                {{Cita pubblicazione|cognome=Kennedy |nome=Edward S. |data=1962 |titolo=Review: ''
                The Observatory in Islam and Its Place in the General History of the Observatory'' by Aydin Sayili |rivista=
                [[Isis (periodico)|Isis]] |volume=53 |numero=2 |pp=237-239 |doi=10.1086/349558 }}
                {{cita libro\n | autore = F. H. Shu\n | titolo = The Physical Universe\n | pubblicazione = University Science Books\n
                | data = 1982\n | città = Mill Valley, California\n | isbn = 0-935702-05-9}}
                {{cita web\n |titolo = Penn State Erie-School of Science-Astronomy and Astrophysics\n |url = http://www.erie.psu.edu/academic/science/degrees/astronomy/astrophysics.htm\n |accesso      = 20 giugno 2007\n |urlmorto     = sì\n |urlarchivio  = https://web.archive.org/web/20071101100832/http://www.erie.psu.edu/academic/science/degrees/astronomy/astrophysics.htm\n |dataarchivio = 1º novembre 2007\n}}
                {{cite web|first1=Peter|last1=Caranicas|access-date=November 7, 2018|title=ILM Launches TV Unit to Serve Episodic and Streaming Content|url=https://variety.com/2018/artisans/news/george-lucas-star-wars-ilm-launches-tv-unit-1203022007/|website=Variety|date=November 7, 2018}}
                """
        wikicode = mwparserfromhell.parse(mwtext)
        for tpl in wikicode.filter_templates():
            parsed = parse_citation_template(tpl, 'it')
            print(parsed)
            # All templates in this example are citation templates
            self.assertIsInstance(parsed, dict)
            self.assertNotEqual(parsed, "{}")

    def test_translate_nl(self):
        mwtext = """
                {{Citeer web|url=https://eurovision.tv/about/rules|titel=The Rules of the Contest|taal=en|werk=Eurovision.tv|archiefurl=https://web.archive.org/web/20220716003114/https://eurovision.tv/about/rules|archiefdatum=16 juli 2022}}
                {{Citeer boek | achternaam = D Hillenius ea | titel = Spectrum Dieren Encyclopedie Deel 7 | pagina's = Pagina 2312 | datum = 1971 | uitgever = Uitgeverij Het Spectrum| ISBN = 90 274 2097 1}}
                {{citeer journal | auteur = Higgs PG | title = RNA secondary structure: physical and computational aspects | url = https://archive.org/details/sim_quarterly-reviews-of-biophysics_2000-08_33_3/page/199 | journal = Quarterly Reviews of Biophysics | volume = 33 | issue = 3 | pages = 199–253 | date = 2000 | pmid = 11191843 | doi = 10.1017/S0033583500003620 }}
                """
        wikicode = mwparserfromhell.parse(mwtext)
        for tpl in wikicode.filter_templates():
            parsed = parse_citation_template(tpl, 'nl')
            print(parsed)
            # All templates in this example are citation templates
            self.assertIsInstance(parsed, dict)
            self.assertNotEqual(parsed, "{}")

    def test_translate_pl(self):
        mwtext = """
            {{cytuj stronę | url = http://www.dreamcharleston.com/charleston-history.html | tytuł= An Overview History of Charleston S.C.| opublikowany = Dream, Charleston, SC | język = en | data dostępu = 2018-01-18}}
            {{cytuj | autor = Mateusz Tałanda | tytuł = Cretaceous roots of the amphisbaenian lizards | czasopismo = Zoologica Scripta | data = 2016-01 | data dostępu = 2022-07-06 | wolumin = 45 | numer = 1 | s = 1–8 | doi = 10.1111/zsc.12138 | język = en | dostęp = z}}
            {{Cytuj pismo |tytuł=Low-Level Nuclear Activity in Nearby Spiral Galaxies |autor=Himel Ghosh, Smita Mathur, Fabrizio Fiore, Laura Ferrarese |arxiv=0801.4382v2 |czasopismo=arXiv + The Astrophysical Journal  |data=2008-06-24 |język=en |data dostępu=2012-12-13 |doi=10.1086/591508 |wolumin=687 |wydanie=1 |strony=216-229}}
            {{cytuj książkę\n | autor= Kasjusz Dion Kokcejanus\n | autor link= Kasjusz Dion\n | tytuł= Historia\n | url=  http://penelope.uchicago.edu/Thayer/E/Roman/Texts/Cassius_Dio/home.html\n | rozdział= Księga LIII 1; LIV 6\n}}
                """
        wikicode = mwparserfromhell.parse(mwtext)
        for tpl in wikicode.filter_templates():
            parsed = parse_citation_template(tpl, 'pl')
            print(parsed)
            # All templates in this example are citation templates
            self.assertIsInstance(parsed, dict)
            self.assertNotEqual(parsed, "{}")

    def test_translate_fr(self):
        mwtext = """
            {{Article|auteur1=Walter Moser|titre=« Puissance baroque » dans les nouveaux médias. À propos de Prospero’s Books de Peter Greenaway|périodique=Cinémas|volume=10|numéro=2-3|date=printemps 2000|lire en ligne=https://doi.org/10.7202/024815ar|pages=39–63}}
            {{article|prénom1=|nom1=|url=http://www.journaldesfemmes.com/loisirs/cinema/1740005-palmares-globes-de-cristal-2017-ceremonie/|titre=Globes de Cristal 2017 : le palmarès complet|consulté le=31 janvier 17|périodique=Le Journal des femmes|jour=31|mois=janvier|année=2017}}
            {{lien web|url=https://news.google.com/newspapers?nid=1310&dat=19580426&id=sMgUAAAAIBAJ&pg=3303,4437749|titre=Little America Will Float Away on an Iceberg|site=Eugene Register-Guard|date=April 1958|consulté le=14 décembre 2009}}
            {{article|titre=Mythopoïèse et souffrance familiale|auteur=Evelyn Granjon|année=2000|pages=13-22|périodique=Le Divan familial|numéro=4}}
            {{Ouvrage|auteur1=Sous la direction d'Emmanuelle Brugerolles|titre=Rembrandt et son entourage, Carnets d'études 23|passage=p. 74-77, Cat. 18|éditeur=Beaux-arts de Paris les éditions|date=2012-2014}}
            {{Ouvrage|auteur1=Emile Wiriot|titre=Le quartier Saint-Jacques et les quartiers voisins|éditeur=Tolra|année=1930|passage=254|lire en ligne=https://gallica.bnf.fr/ark:/12148/bpt6k165711w/f254.image}}
            """
        wikicode = mwparserfromhell.parse(mwtext)
        for tpl in wikicode.filter_templates():
            parsed = parse_citation_template(tpl, 'fr')
            print(parsed)
            # All templates in this example are citation templates
            self.assertIsInstance(parsed, dict)
            self.assertNotEqual(parsed, "{}")


if __name__ == '__main__':
        unittest.main()
