from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
from keras.models import load_model
import logging
import jieba

PROJECT_HOME = 'c:///users/natal/PycharmProjects/cite-classifications-wiki/'
ext = "_xh"


class WikiSentences:
    def __init__(self, wiki_dump_path, lang):
        logging.info('Parsing wiki corpus')
        self.wiki = WikiCorpus(wiki_dump_path)
        self.lang = lang

    def __iter__(self):
        for sentence in self.wiki.get_texts():
            if self.lang == 'zh':
                yield list(jieba.cut(''.join(sentence), cut_all=False))
            else:
                yield list(sentence)


if __name__ == '__main__':
    args = {
        'model'   : 'fasttext',
        'lang'    : 'xh',
        'path_in' : PROJECT_HOME + 'data/dumps/xhwiki-20221001-pages-articles-multistream.xml.bz2',
        'path_out': PROJECT_HOME + 'data/model/fasttext' + ext + '.model'
    }
    # wiki_sentences = WikiSentences(args["path_in"], args["lang"])
    #
    # if args["model"] == 'word2vec':
    #     model = Word2Vec(wiki_sentences, sg=1, hs=1)
    # elif args["model"] == 'fasttext':
    #     model = FastText(wiki_sentences, sg=1, hs=1)
    #
    # model.save(args["path_out"])
    FASTTEXT_MODEL = '../../data/model/xh_wiki_fasttext.txt'
    # FASTTEXT_MODEL = 'file://' + PROJECT_HOME + 'data/model/xh_wiki_fasttext.txt.wv.vectors_ngrams.npy'
    model = FastText.load(FASTTEXT_MODEL)
