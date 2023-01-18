from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
import jieba


class WikiSentences:
    def __init__(self, wiki_dump_path, lang):
        self.wiki = WikiCorpus(wiki_dump_path)
        self.lang = lang

    def __iter__(self):
        for sentence in self.wiki.get_texts():
            if self.lang == 'zh':
                yield list(jieba.cut(''.join(sentence), cut_all=False))
            else:
                yield list(sentence)


def train_fasttext_model(file_in, file_out, lang):
    model = FastText(WikiSentences(file_in, lang), sg=1, hs=1)
    model.save(file_out)


def train_word2vec_model(file_in, file_out, lang):
    model = Word2Vec(WikiSentences(file_in, lang), sg=1, hs=1)
    model.save(file_out)
