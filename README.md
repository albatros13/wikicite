# Extraction and classification of references from Wikipedia articles in multiple languages.

This is the experimental code for Wikipedia citation extraction in multiple languages.
Given a Wikipedia dump, the code retrieves citations and harmonizes citation templates (structure citations using the same schema).  
After extraction, selected citation templates are translated from original language to English. 

For example, an Italian Wikipedia citation
'{{cita libro | autore = Barth, F. | titolo = Ethnic groups and boundaries: The social organization of culture differences | url = https://archive.org/details/ethnicgroupsboun0000unse | lingua = en | editore = Little Brown & Co. | città = Boston | anno = 1969}}'
is parsed, translated, and represented as:
{'Authors': [{'last': 'Barth, F.'}], 'CitationClass': 'cite book', 'PublicationPlace': 'Boston', 'PublisherName': 'Little Brown & Co.', 'Title': 'Ethnic groups and boundaries: The social organization of culture differences', 'URL': 'https://archive.org/details/ethnicgroupsboun0000unse'}

The rest of the pipeline is similar to the implementation of English Wikipedia citation extraction (https://github.com/albatros13/wikicite). 
