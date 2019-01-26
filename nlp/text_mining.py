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
from sqlalchemy import create_engine

logger = logging.getLogger('text_mining')
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

path = os.path.dirname(os.path.abspath(__file__))
m = Morfeusz()

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

def move_csv_to_db():
    root_path = os.path.dirname(os.path.abspath(__file__))
    db_file = os.path.abspath(os.path.join(root_path, '..', 'data', 'personaldev.sqlite'))
    engine = create_engine('sqlite:///' + db_file, echo=False)
    is_first = True
    for dirpath, dnames, fnames in os.walk(os.path.join(root_path, 'data')):
        for file in fnames:
            if file.endswith(".csv"):
                data_pd = pd.read_csv(os.path.join(path, 'data', file), sep=';', encoding='utf-8', quotechar='"')
                data_pd.to_sql('pracujpl', con=engine, if_exists='replace' if is_first else 'append', index_label='id')
                logger.info("DB count: " + str(engine.execute("SELECT count(*) FROM pracujpl").fetchone()[0]))


def clear_txt(txt):
    txt = txt.lower()
    txt = re.sub("[\(\):;,\.\?\-_–‒■‘'\{\}\*@\!\+0-9]+", ' ', txt)
    txt = re.sub('\s+', ' ', txt)
    return txt


def filter_stopword_phrases(phrases_csv_file_path):
    working_dir = create_working_dir()
    phrases_data = pd.read_csv(phrases_csv_file_path, sep=';', encoding='utf-8', quotechar='"')
    stopwords = get_stopwords()
    phrases_data = phrases_data[phrases_data.apply(lambda x: x['Phrase'] not in stopwords, axis=1)]
    phrases_data.sort_values(by=['DF'], ascending=False).to_csv(
        os.path.join(path, working_dir, 'phrases_with_stats__no_stopwords.csv'),
        sep=';', encoding='utf-8', mode='w', quotechar='"', line_terminator='\n')


def create_working_dir():
    working_dir = 'text_mining__{}'.format(time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()))

    if not os.path.exists(os.path.join(path, working_dir)):
        os.makedirs(os.path.join(path, working_dir))
    return os.path.join(path, working_dir)


def find_phrases_and_stats(path, use_stemmed_dataFrom_path=None, desired_lang='pl'):
    working_dir = create_working_dir()
    error_words_pd = pd.read_csv(os.path.join(path, 'error_phrase_replacements.csv'), sep=';', encoding='utf-8',
                                 quotechar='"')
    error_words = dict(zip(error_words_pd['a'].values, error_words_pd['b'].values))
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
                    txt = ' '.join([word if word not in error_words else error_words[word] for word in txt.split()])
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


def fix_broken_phrases():
    m = Morfeusz()
    info = m.analyse('bittttt')

    test = m.analyse('prawidłowościzarządzenie')[0][2][2]
    phrases_csv_file_path = os.path.join(path, 'phrases_with_stats_45697_pl.csv')
    # phrases_csv_file_path = os.path.join(path, 'phrases_test.csv')
    phrases_data = pd.read_csv(phrases_csv_file_path, sep=';', encoding='utf-8', quotechar='"')
    phrases_data = set(phrases_data[phrases_data.apply(
        lambda x: x['Word_count'] == 1 and isinstance(x['Phrase'], str) and len(x['Phrase']) > 0 and
                  m.analyse(x['Phrase'])[0][2][2] == 'ign', axis=1)]['Phrase'].values)
    replacements = {}
    for error_phrase in phrases_data:
        replacements[error_phrase] = error_phrase
        found = False
        for i in range(1, len(error_phrase)):
            a = error_phrase[0:i]
            b = error_phrase[i:]
            if m.analyse(a)[0][2][2] != 'ign' and m.analyse(b)[0][2][2] != 'ign':
                replacements[error_phrase] = ' '.join((a, b))
                found = True
        if not found:
            for i in range(1, len(error_phrase)):
                a = error_phrase[0:i]
                b = error_phrase[i:]
                if m.analyse(b)[0][2][2] != 'ign':
                    replacements[error_phrase] = ' '.join((a, b))
                    break

    replacements_pd = pd.DataFrame(list(zip(list(replacements.keys()), list(replacements.values()))),
                                   columns=['a', 'b'])
    replacements_pd.to_csv(os.path.join(path, 'error_phrase_replacements.csv'), sep=';',
                           encoding='utf-8', mode='w', quotechar='"', line_terminator='\n')


if __name__ == "__main__":
    # https://github.com/nikita-moor/morfeusz-wrapper
    logger.info("START")
    move_csv_to_db()
    # path = os.path.dirname(os.path.abspath(__file__))
    # working_dir = find_phrases_and_stats(path)
    # # working_dir = find_phrases_and_stats(path, use_stemmed_dataFrom_path=os.path.join(path, 'text_mining__2019_01_16__19_39_36', 'full_data_with_stem_column.csv'))
    # logger.info("Data saved in '{}'".format(working_dir))
    logger.info("DONE")
