import os
import re
import gensim
import pickle
import time
from gensim.corpora import Dictionary
import pandas as pd
from morfeusz2 import Morfeusz
from langdetect import detect
from langdetect.detector import LangDetectException
import logging
from stopwords import get_stopwords

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
    # fivegram = gensim.models.Phrases(fourgram[corpus], min_count=30, threshold=4.0)
    # fivegram = gensim.models.Phrases(fourgram[corpus], min_count=50, threshold=0.5, scoring='npmi')
    # sixgram = gensim.models.Phrases(fivegram[corpus], min_count=30, threshold=4.0)
    # sixgram = gensim.models.Phrases(fivegram[corpus], min_count=50, threshold=0.5, scoring='npmi')

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    fourgram_mod = gensim.models.phrases.Phraser(fourgram)
    # fivegram_mod = gensim.models.phrases.Phraser(fivegram)
    # sixgram_mod = gensim.models.phrases.Phraser(sixgram)

    return [
        # sixgram_mod[
        #     fivegram_mod[
        fourgram_mod[
            trigram_mod[
                bigram_mod[data_words]
            ]
        ]
        # ]
        # ]
        for data_words in corpus]


def read_full_data():
    path = os.path.dirname(os.path.abspath(__file__))
    data_frames = []
    for dirpath, dnames, fnames in os.walk(os.path.join(path, 'data')):
        for file in fnames:
            if file.endswith(".csv"):
                data_frames.append(
                    pd.read_csv(os.path.join(path, 'data', file), sep=';', encoding='utf-8', quotechar='"'))
    data = pd.concat(data_frames, ignore_index=True)
    return data


def clear_txt(txt):
    txt = txt.lower()
    txt = re.sub('[\(\):;,\.\?\-_–‒‘\{\}\*@\!\+0-9]+', ' ', txt)
    txt = re.sub('\s+', ' ', txt)
    return txt


def find_phrases_and_stats(path, use_stemmed_dataFrom_path=None, desired_lang='pl'):
    working_dir = 'text_mining__{}'.format(time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()))

    if not os.path.exists(os.path.join(path, working_dir)):
        os.makedirs(os.path.join(path, working_dir))
    if use_stemmed_dataFrom_path is None:
        m = Morfeusz()
        data = read_full_data()
        data_gen = data.iterrows()
        logger.info("Data read")
        corpus = []
        data_with_stem = []
        i = 0
        data_length = len(data)
        for data_serie in data_gen:
            txt = clear_txt(data_serie[1]['content'])
            try:
                lang = detect(txt);
                if lang == desired_lang and len(txt) > 100:
                    stem = [m.analyse(word)[0][2][1] for word in txt.split()]
                    # stem = [a[2][1][0] for a in m.analyse(txt)]
                    corpus.append(stem)
                    data_with_stem.append((data_serie[1]['year'], data_serie[1]['month'], data_serie[1]['title'],
                                           data_serie[1]['location'], txt, ' '.join(stem)))
                    i += 1
                    if i % 10000 == 0:
                        logger.info("Done {0} out of {1} ({2}%)".format(str(i), str(data_length),
                                                                        str(round((i / data_length) * 100))))
            except LangDetectException:
                pass
        logger.info('corpus read')
        full_data_pd = pd.DataFrame(data_with_stem, columns=['year', 'month', 'title', 'location', 'content', 'stem'])
        full_data_pd.to_csv(os.path.join(path, working_dir, 'full_data_with_stem_column.csv'), sep=';',
                            encoding='utf-8', mode='w', quotechar='"', line_terminator='\n')
    else:
        logger.info("Reading stemmed data from CSV")
        full_data_pd = pd.read_csv(use_stemmed_dataFrom_path, sep=';', encoding='utf-8', quotechar='"')
        corpus = [sentence.split() for sentence in full_data_pd['stem'].values]
        del full_data_pd
    logger.info('Applying phrases')
    phrases = applyPhrases(corpus)
    # pickle.dump(phrases, open(os.path.join(path, working_dir, 'raw_phrases_{}.pickle'.format(desired_lang)), 'wb'))
    logger.info('Phrases applied')
    dct = Dictionary(phrases)

    logger.info('Phrases stats...')
    phrases_set = set([phrase for sent in phrases for phrase in sent if len(phrase) > 1])
    stopwords = get_stopwords()
    phrases_set = [phrase for phrase in phrases_set if phrase not in stopwords]
    del phrases
    dfs = [dct.dfs[dct.token2id[s]] for s in phrases_set]
    word_count = [len(s.split('_')) for s in phrases_set]

    phrase_stats = list(zip([s.replace('_', ' ') for s in phrases_set], word_count, dfs))
    phrases_pd = pd.DataFrame(phrase_stats, columns=['Phrase', 'Word count', 'DF'])
    phrases_pd.sort_values(by=['DF'], ascending=False).to_csv(
        os.path.join(path, working_dir, 'phrases_with_stats_{0}_{1}.csv'.format(len(phrases_set), desired_lang)),
        sep=';', encoding='utf-8', mode='w', quotechar='"', line_terminator='\n')
    return working_dir


if __name__ == "__main__":
    m = Morfeusz()
    info = m.analyse('bit')
    # https://github.com/nikita-moor/morfeusz-wrapper
    logger.info("START")
    path = os.path.dirname(os.path.abspath(__file__))
    working_dir = find_phrases_and_stats(path, use_stemmed_dataFrom_path=os.path.join(path, 'text_mining__2019_01_16__19_39_36', 'full_data_with_stem_column.csv'))
    logger.info("Data saved in '{}'".format(working_dir))
    logger.info("DONE")
