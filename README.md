# Extraction and classification of references from Wikipedia articles.

Refactoring of experimental code from https://github.com/Harshdeep1996/cite-classifications-wiki to an automated 
language-agnostic pipeline.

Folder "scripts" contains revised scripts that run on Python 3 in the environment with up-to-date dependencies 
(see requirements.txt for relevant module versions). 
 - process_data.py combines all original scripts

Scripts can be executed locally or in a cloud-based environment (tested on a GCloud dataproc cluster).


## Configuration

- Set PROJECT_HOME to point to your home location, 
- Set INPUT_DATA to the corresponding Wikipedia dump.
We also use a constant "ext" to indicate which language we are working with, it is used as prefix for the pipeline 
output files. 

