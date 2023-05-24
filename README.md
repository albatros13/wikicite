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

- Set PROJECT_HOME to point to your home location, e.g., GCloud bucket `gs://wikicite-1/`. 
- Set INPUT_DIR to the relative path to the directory containing Wikipedia dumps for parsing, e.g., `data/dumps`.
- Set INPUT_OUT to the output folder, e.g., `data/`. 

We use constant `ext` to indicate which language we are working with, it is added as prefix to output file names. 

### Output  
Several sub-directories are used for intermediate result output: 

  1. content   - contains extracted citations [id, 'title' -> 'page_title', 'revision.text' -> 'content']
  2. citations - contains extracted citations with identifiers ['id', 'title', 'revision.text' -> 'content', 'revision.id' -> 'r_id', 'revision.parentid' -> r_parentid, 'citations', 'sections', 'type_of_citation'] 
  3. separated - contains citations with generic template features, i.e., 
        ['Degree', 'City', 'SeriesNumber', 'AccessDate, 'Chapter', 'PublisherName', 'Format', 'Title',
            'URL', 'Series', 'Authors', 'ID_list', 'Encyclopedia', 'Periodical', 'PublicationPlace', 'Date', 
            'Edition', 'Pages', 'Chron', 'Issue', 'Volume', 'TitleType']
  4. base      - contains citations with context features ['id', 'page_title', 'citations_features'].  
  5. features  - complete dataset: merges generic template features with context features, fields ['total_words', 'neighboring_words', 'neigboring_tags] 
     added to the dataframe at step 3. 
  6. features_ids  - filtered dataset with non-empty identifiers (non-empty 'ID_list')
  
### Labelled dataset   
  1. book_journal  - part of the dataset with ids that we recognise as books or journals
  2. news          - citations that we recognise as web citations (based on news agency top domains)   
  3. news/features - part of the dataset that we recognise as web citations (as in news plus auxiliary features) 

## Running data processing script on GCloud
  The script can be executed on PySpark as described S. Harshdeep in the original project wiki.
  Here we provide instructions for configuring a Google Cloud Dataproc cluster for dataset processing.

  1. Create a project following generic instructions as in https://developers.google.com/workspace/guides/create-project. 
  2. Create a storage bucket https://cloud.google.com/storage/docs/creating-buckets.
  3. Create a data folder and download a chosen dump with Wikipedia articles https://dumps.wikimedia.org/wikidatawiki/. 
     Currently, only English-based citation templates are supported. Support for templates in other languages is work in progress.
  4. Create a single node dataproc cluster with enabled JupyterLab component. Configure node to have sufficient memory 
     to process the chosen dump, e.g., we recommend E2-highmem-32 node providing 16CPU/128GB for full English dump.
    ![Sample cluster configuration](images/cluster-config.png)
     
<small>
    Resource availability depends on chosen region, hence you may need to explore several regions to find out which one supports 
     the required configuration. We did not find instructions on how to install custom libraries on workers in 
     1-master + N-workers configurations as the virtual environment configuration examples show only cases with pip and conda libraries 
     which are downloaded from public repositories during cluster creation.    
</small>
     
  5. Open the configuration of the created cluster on the dashboard and access its JupyterLab via the tab "Web Interfaces".
     Upload requirements.txt and zipped libraries wikiciteparser and mwparserfromhell 
     to the folder `home/[user_name]`. Install required dependencies, unzip and install the libraries:
```
     pip install -r requirements.txt
     cd wikiciteparser
     python setup.py install
     cd ../mwparserfromhell
     python setup.py install
```
  6. Download nltk resources where it can be seen by the running pipeline, i.e.,
    
```
    import nltk
    nltk.download('popular',  download_dir='home/')
```
     
  7. Configure a PySpark job to run the process_data.py script on the chosen cluster providing a path to it in the field 
     "Main python file", e.g., `gs://wikicite-1/gscripts/process_data.py`. 
     Upload a spark-xml module to your bucket and provide path to it in jar-files, e.g., 
     `gs://wikicite-1/jars/spark-xml_2.12-0.16.0.jar`  
    ![Job configuration](images/cluster-job.png)
     
  Execute the job (may take up to several days for large files).
     