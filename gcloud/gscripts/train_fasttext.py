from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
import os


class WikiSentences:
    def __init__(self, wiki_dump_path, lang):
        self.wiki = WikiCorpus(wiki_dump_path)
        self.lang = lang

    def __iter__(self):
        for sentence in self.wiki.get_texts():
            yield list(sentence)


def train_fasttext_model(file_in, file_out, lang):
    model = FastText(WikiSentences(file_in, lang), sg=1, hs=1)
    model.save(file_out)


def train_word2vec_model(file_in, file_out, lang):
    model = Word2Vec(WikiSentences(file_in, lang), sg=1, hs=1)
    model.save(file_out)


# PROJECT_HOME = 'c:///users/natal/PycharmProjects/cite-classifications-wiki/'
# PROJECT_HOME = "gs://wikicite-1/"
PROJECT_HOME = "/home/natal/"
ext = "en_"
INPUT_DATA = PROJECT_HOME + 'data/dumps/enwiki-20221201-pages-articles-multistream1.xml-p1p41242.bz2'

FASTTEXT_MODEL = '../data/model/{}wiki_fasttext.txt'.format(ext)

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print("Running in folder:", dir_path)

    train_fasttext_model(INPUT_DATA, FASTTEXT_MODEL, ext[0:2])

