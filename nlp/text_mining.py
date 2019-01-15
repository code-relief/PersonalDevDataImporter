import os
import re
import gensim
import pickle
from gensim.corpora import Dictionary
import pandas as pd
from morfeusz2 import Morfeusz
from langdetect import detect
from langdetect.detector import LangDetectException
import logging

logger = logging.getLogger('text_mining')
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

def applyPhrases(corpus):
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(corpus, min_count=100, threshold=10.0)
    # bigram = gensim.models.Phrases(corpus, min_count=100, threshold=0.8, scoring='npmi')
    trigram = gensim.models.Phrases(bigram[corpus], min_count=40, threshold=5.0)
    # trigram = gensim.models.Phrases(bigram[corpus], min_count=50, threshold=0.5, scoring='npmi')
    fourgram = gensim.models.Phrases(trigram[corpus], min_count=30, threshold=4.0)
    # fourgram = gensim.models.Phrases(trigram[corpus], min_count=50, threshold=0.5, scoring='npmi')
    fivegram = gensim.models.Phrases(fourgram[corpus], min_count=30, threshold=4.0)
    # fivegram = gensim.models.Phrases(fourgram[corpus], min_count=50, threshold=0.5, scoring='npmi')
    sixgram = gensim.models.Phrases(fivegram[corpus], min_count=30, threshold=4.0)
    # sixgram = gensim.models.Phrases(fivegram[corpus], min_count=50, threshold=0.5, scoring='npmi')

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    fourgram_mod = gensim.models.phrases.Phraser(fourgram)
    fivegram_mod = gensim.models.phrases.Phraser(fivegram)
    sixgram_mod = gensim.models.phrases.Phraser(sixgram)

    return [
        sixgram_mod[
            fivegram_mod[
                fourgram_mod[
                    trigram_mod[
                        bigram_mod[data_words]
                    ]
                ]
            ]
        ]
        for data_words in corpus]

def read_full_data():
    path = os.path.dirname(os.path.abspath(__file__))
    data_frames = []
    for dirpath, dnames, fnames in os.walk(os.path.join(path, 'data')):
        for file in fnames:
            if file.endswith(".csv"):
                data_frames.append(pd.read_csv(os.path.join(path, 'data', file), sep=';', encoding='utf-8', quotechar='"'))
    data = pd.concat(data_frames, ignore_index=True)
    return data

def preprocess_txt(corpus, lang='pl'):

    lang_corpus = [txt for txt in corpus if len(txt) > 100 and detect(txt) is lang]
    # stemming
    # phrases
    return lang_corpus

def clear_txt(txt):
    txt = txt.lower()
    txt = re.sub('[\(\):;,\.\?\-_\{\}\*@\!\+]+', ' ', txt)
    txt = re.sub('\s+', ' ', txt)
    return txt


if __name__ == "__main__":
    # https://github.com/nikita-moor/morfeusz-wrapper
    logger.info("Start")
    path = os.path.dirname(os.path.abspath(__file__))
    m = Morfeusz()
    res = m.analyse("Idę")
    res_stemmed = res[0][2][1]
    data = read_full_data()
    data_gen = data.iterrows()
    logger.info("Data read")
    corpus = []
    desired_lang = 'pl'
    i = 0
    data_length = len(data)
    for data_serie in data_gen:
        txt = clear_txt(data_serie[1]['content'])
        try:
            lang = detect(txt);
            if lang == desired_lang:
                stem = [a[2][1] for a in m.analyse(txt)]
                corpus.append(stem)
                i += 1
                if i%1000 == 0:
                    logger.info("Done {0} out of {1}".format(str(i), str(data_length)))
        except LangDetectException:
            pass
    print('corpus read')
    pickle.dump(corpus, os.path.join(path, 'corpus_stemmed.pickle'))
    subjects = applyPhrases(preprocess_txt(corpus, lang='pl'))
    pickle.dump(subjects, os.path.join(path, 'phrases.pickle'))
    dct = Dictionary(subjects)
    print('phrases applied')

    subject_set = list(set([s[0] for s in subjects if len(s) == 1]))
    dfs = [dct.dfs[dct.token2id[s]] for s in subject_set]
    word_count = [len(s.split('_')) for s in subject_set]

    result = zip([s.replace('_', ' ') for s in subject_set], word_count, dfs)