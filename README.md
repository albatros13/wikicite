# Extraction and classification of references from Wikipedia articles.

Refactoring of experimental code from https://github.com/Harshdeep1996/cite-classifications-wiki to an automated 
language-agnostic pipeline

Folder "scripts" contains revised scripts that run on Python 3 in the environment with up-to-date dependencies 
(see requirements.txt for relevant module versions). 

Folder "gscripts" contains a file with all scripts together for easier execution of the whole pipeline 
locally or on a cloud-based environment (tested on a GCloud dataproc cluster).
Set the PROJECT_HOME folder to point to your home location, and INPUT_DATA to the corresponding Wikipedia dump.
We also use a constant "ext" to indicate which language we are working with, it is used as prefix for the pipeline 
output files. 

