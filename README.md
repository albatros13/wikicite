# Extraction and classification of references from Wikipedia articles.

This is the refactored and generalized code for Wikipedia citation extraction originally developed by Harshdeep et al. 
(https://github.com/Harshdeep1996/cite-classifications-wiki). 

Folder "scripts" contains scripts that run on Python 3 in the environment with up-to-date dependencies 
(see requirements.txt for relevant module versions). 
 - process_data.py combines all data processing functions related to citation extraction and dataset preparation
 - train_citation_classifier.py deals with labelled dataset preparation and classifier training
 - predict_citations.py allows users to classify extracted citations to chosen categories (book, journal, or web currently).

Scripts can be executed locally or in a cloud-based environment (tested on a GCloud dataproc cluster).

## Configuration

- Set PROJECT_HOME to point to your home location, e.g., "gs://wikicite-1/" (GCloud bucket). 
- Set INPUT_DIR to the relative path to the directory containing Wikipedia dumps for parsing, e.g., "data/en_parts/dumps".
- Set INPUT_OUT to the output folder, e.g., "data/en_parts". 

We also use a constant "ext" to indicate which language we are working with, it is used as prefix for the pipeline 
output files. 

### Output  
Several sub-directories are used for intermediate result output: 

  - citations - extracted citations ['id', 'title', 'revision.text' -> 'content', 'revision.id' -> 'r_id', 'revision.parentid' -> r_parentid, 'citations', 'sections', 'tyep_of_citation'] 
  - separated - parsed citations with fields like in 'citations' plus the generic template fields, i.e., 
        ['Degree', 'City', 'SeriesNumber', 'AccessDate, 'Chapter', 'PublisherName', 'Format', 'Title',
            'URL', 'Series', 'Authors', 'ID_list', 'Encyclopedia', 'Periodical', 'PublicationPlace', 'Date', 
            'Edition', 'Pages', 'Chron', 'Issue', 'Volume', 'TitleType']
  - content   - extracted citation content [id, 'title' -> 'page_title', 'revision.text' -> 'content']
  - base      - processed citation content to obtain auxiliary features ['id', 'page_title', 'citations_features'].  
  - features  - complete dataset: merged publication template features and auxiliary features including ['total_words', 'neighboring_words', 'neigboring_tags]  
  - features_ids  - part of the dataset with non-empty identifiers (ID_list)
  
### Labelled dataset  
  - book_journal  - part of the dataset with ids that we recognise as books or journals
  - news          - citations that we recognise as web citations (based on news agency top domains)   
  - news/features - part of the dataset that we recognise as web citations (as in news plus auxiliary features) 
