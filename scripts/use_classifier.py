from local import PROJECT_HOME
from ml.train_gensim_model import train_fasttext_model
from predict_citations import predict_citations

ext = "xh_"
FASTTEXT_MODEL = '../data/model/{}wiki_fasttext.txt'.format(ext)
# INPUT_DATA = PROJECT_HOME + 'data/dumps/enwiki-20221201-pages-articles-multistream1.xml-p1p41242.bz2'
INPUT_DATA = PROJECT_HOME + 'data/dumps/xhwiki-20221001-pages-articles-multistream.xml.bz2'

if __name__ == '__main__':
    # Train FastText model before calling classifier
    # train_fasttext_model(INPUT_DATA, FASTTEXT_MODEL, ext[0:2])

    # Execute train_citation_classifier before calling classifier

    predict_citations(PROJECT_HOME, ext)