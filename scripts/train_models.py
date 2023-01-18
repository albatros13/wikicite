from local import PROJECT_HOME
from ml.train_gensim_model import train_fasttext_model

ext = "_en"
INPUT_DATA = PROJECT_HOME + 'data/dumps/enwiki-20221201-pages-articles-multistream1.xml-p1p41242.bz2'
FASTTEXT_MODEL = '../data/model/wiki_fasttext{}.txt'.format(ext)

if __name__ == '__main__':
    train_fasttext_model(INPUT_DATA, FASTTEXT_MODEL, ext[1:])
