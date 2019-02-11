import os
import re
import gensim
import pickle
import time
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import LdaModel
import pandas as pd
from morfeusz2 import Morfeusz
from langdetect import detect
from langdetect.detector import LangDetectException
import logging
from stopwords import get_stopwords
from sqlalchemy import create_engine
from gensim.models import CoherenceModel
from gensim.summarization import keywords
from gensim.summarization.summarizer import summarize
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# %matplotlib inline

logger = logging.getLogger('text_mining')
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

path = os.path.dirname(os.path.abspath(__file__))
m = Morfeusz()
os.environ['MALLET_HOME'] = 'C:\\Projects\\miscellaneous\\mallet-2.0.8'
mallet_path = 'C:\\Projects\\miscellaneous\\mallet-2.0.8\\bin\\mallet'


def applyPhrases(corpus):
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(corpus, min_count=50, threshold=4.0)
    # bigram = gensim.models.Phrases(corpus, min_count=100, threshold=0.8, scoring='npmi')
    trigram = gensim.models.Phrases(bigram[corpus], min_count=40, threshold=4.0)
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


def move_csv_to_db():
    engine = create_sqlite_engine()
    is_first = True
    for dirpath, dnames, fnames in os.walk(os.path.join(path, 'data')):
        for file in fnames:
            if file.endswith(".csv"):
                data_pd = pd.read_csv(os.path.join(path, 'data', file), sep=';', encoding='utf-8', quotechar='"')
                data_pd.to_sql('pracujpl', con=engine, if_exists='replace' if is_first else 'append', index_label='id')
                logger.info("DB count: " + str(engine.execute("SELECT count(*) FROM pracujpl").fetchone()[0]))
                is_first = False


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


def preprocess_corpus_and_find_phrases_and_stats(preprocess_content=True, desired_lang='pl'):
    error_phrase_replacements = {}
    if preprocess_content:
        m = Morfeusz()
        engine = create_sqlite_engine()
        data = engine.execute(
            "SELECT id, content FROM pracujpl WHERE content_fixed is null and lang is null").fetchall()
        logger.info("Data read")
        i = 0
        data_length = len(data)
        for row in data:
            txt = row[1]
            id = row[0]
            try:
                lang = detect(txt);
                if lang == desired_lang and len(txt) > 100:
                    txt = ' '.join(
                        [word if word not in error_phrase_replacements else error_phrase_replacements[word] for word in
                         txt.lower().split()])
                    txt = re.sub("[';:!\\.,\\(\\)]", ' ', txt)
                    repl_size = len(error_phrase_replacements.keys())
                    stem = [extract_stem(m.analyse(word), m, error_phrase_replacements) for word in txt.split()]
                    if repl_size < len(error_phrase_replacements.keys()):
                        txt = ' '.join(
                            [word if word not in error_phrase_replacements else error_phrase_replacements[word] for word
                             in txt.split()])
                    i += 1
                    if i % 10000 == 0:
                        logger.info("Done {0} out of {1} ({2}%)".format(str(i), str(data_length),
                                                                        str(round((i / data_length) * 100))))
                    engine.execute(
                        "UPDATE pracujpl SET content='{0}', stem='{1}', lang='{2}', content_fixed=1 WHERE id={3}".format(
                            txt, ' '.join(stem), lang, id))
                else:
                    engine.execute("UPDATE pracujpl SET lang='{0}' WHERE id={1}".format(lang, id))
            except LangDetectException:
                engine.execute("UPDATE pracujpl SET content_fixed=0 WHERE id={}".format(id))
        logger.info('corpus transformed')
        del data
    corpus = [stem[0].split() for stem in engine.execute(
        "SELECT stem FROM pracujpl WHERE lang='{}' AND content_fixed=1".format(desired_lang)).fetchall()]
    logger.info('Applying phrases')
    phrases = applyPhrases(corpus)

    # pickle.dump(phrases, open(os.path.join(path, working_dir, 'raw_phrases_{}.pickle'.format(desired_lang)), 'wb'))
    logger.info('Phrases applied')
    dct = Dictionary(phrases)
    bow = [dct.doc2bow(phrase) for phrase in phrases]
    tfidf_model = TfidfModel(bow)

    logger.info('Phrases stats...')
    phrases_set = list(set([phrase for sent in phrases for phrase in sent if len(phrase) > 1]))
    stopwords = get_stopwords()
    phrases_set = [phrase for phrase in phrases_set if phrase not in stopwords]
    del phrases
    dfs = [dct.dfs[dct.token2id[s]] for s in phrases_set]
    tfidfs = [tfidf_model[dct.doc2bow([s])] for s in phrases_set]
    word_count = [len(s.split('_')) for s in phrases_set]
    pos_data = [extract_pos(phrase.split("_")) for phrase in phrases_set]
    pos_elements, pos_annotations = zip(*pos_data)

    # phrase_stats = list(zip([s.replace('_', ' ') for s in phrases_set], pos_elements, pos_annotations, word_count, dfs, tfidfs))
    phrase_stats = list(zip([s.replace('_', ' ') for s in phrases_set], pos_elements, pos_annotations, word_count, dfs))
    # phrases_pd = pd.DataFrame(phrase_stats, columns=['phrase', 'pos', 'word_count', 'global_df', 'global_tfidf'])
    phrases_pd = pd.DataFrame(phrase_stats,
                              columns=['phrase', 'pos_element', 'pos_annotation', 'word_count', 'global_df'])
    phrases_pd.to_sql('pracujpl_phrases', con=engine, if_exists='replace', index_label='id')


def extract_main_job_titles():
    engine = create_sqlite_engine()
    job_titles = engine.execute(
        "select lower(offerData_jobTitle) as offerData_jobTitle from pracujpl where offerData_jobTitle is not Null and lang='pl'").fetchall()
    job_titles = [title[0].split() for title in job_titles]
    logger.info('Data read')
    phrases = applyPhrases(job_titles)
    logger.info('Phrases applied')
    dct = Dictionary(phrases)
    logger.info('DICT created')
    phrases_set = list(set([phrase for sent in phrases for phrase in sent if len(phrase) > 1]))
    dfs = [dct.dfs[dct.token2id[s]] for s in phrases_set]
    word_count = [len(s.split('_')) for s in phrases_set]
    logger.info('Stats canculated')
    phrases_set = [s.replace('_', ' ') for s in phrases_set]
    pos_data = [extract_pos(phrase.split()) for phrase in phrases_set]
    pos_elements, pos_annotations = zip(*pos_data)

    main_job_titles = pd.DataFrame(list(zip(phrases_set, pos_elements, pos_annotations, word_count, dfs)),
                                   columns=['job_title', 'pos_element', 'pos_annotation', 'word_count', 'df'])
    # print(main_job_titles.sort_values(by=['df'], ascending=False))
    main_job_titles.to_sql('job_titles', con=engine, if_exists='replace', index_label='id')


def extract_pos(words):
    pos = []
    annotations = []
    for word in words:
        morf_data = m.analyse(word)
        for word_morf in morf_data:
            pos.append(word_morf[2][2])
            annotations.append('' if len(word_morf[2][3]) == 0 else word_morf[2][3][0])
            break
    return ','.join(pos), ','.join(annotations)


def extract_stem(morf_data, m, replacements):
    stem = ''
    error = []
    for word_morf in morf_data:
        stem = word_morf[2][1]
        if word_morf[2][2] == 'ign':
            fix_broken_phrase(stem, m, replacements)
        break
    return stem


def fix_broken_phrase(error_phrase, m, replacements):
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


def create_sqlite_engine():
    root_path = os.path.dirname(os.path.abspath(__file__))
    db_file = os.path.abspath(os.path.join(root_path, '..', 'data', 'personaldev.sqlite'))
    return create_engine('sqlite:///' + db_file, echo=False)


def perform_topic_modelling():
    sql_programming = "select ppl.stem from pracujpl ppl inner join job_titles jt on jt.pos_annotation='nazwa języka programowania' and ppl.offerData_jobTitle LIKE '%'||jt.job_title||'%' where ppl.stem is not null"
    sql_salles = "select ppl.stem from pracujpl ppl where offerData_jobTitle LIKE '%sprzedawca%' and ppl.stem is not null"
    sql_drivers = "select ppl.stem from pracujpl ppl where offerData_jobTitle LIKE '%kierowca międzynarodowy%' and ppl.stem is not null"
    engine = create_sqlite_engine()
    corpus = engine.execute(sql_drivers).fetchall()
    logger.info("Data read. Corpus size: {}".format(len(corpus)))
    corpus = [txt[0].split() for txt in corpus]
    corpus = applyPhrases(corpus)
    logger.info("Phrases applied")
    stopwords = get_stopwords()
    corpus = [[word for word in doc if word not in stopwords] for doc in corpus]
    logger.info("Stopwords removed")
    dict = Dictionary(corpus)
    logger.info("Dict created")
    corpus_bow = [dict.doc2bow(text) for text in corpus]
    logger.info("BOW calculated")
    coherences = {}
    for topics in (2, 4, 6, 8, 10, 12, 15):
        logger.info("Training TFIDF Mallet LDA for {} topics".format(topics))
        ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus_bow, num_topics=topics, id2word=dict,
                                                     workers=8, iterations=800)
        # Show Topics
        print(ldamallet.show_topics(num_topics=-1, num_words=20))

        coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=corpus, dictionary=dict, coherence='c_v')
        coherence_ldamallet = coherence_model_ldamallet.get_coherence()
        print('Topics: ', topics)
        print('\nCoherence Score: ', coherence_ldamallet)
        coherences[topics] = coherence_ldamallet

    print(coherences)

    # lda = LdaModel(corpus=corpus_bow, id2word=dict, num_topics=5, passes=10, alpha='auto', update_every=50,
    #                chunksize=100, random_state=100)
    # logger.info("LDA applied")
    # top_words = [[word for word, prob in lda.show_topic(topicno, topn=20)] for topicno in range(lda.num_topics)]
    # print(top_words)
    # Compute Perplexity
    # print('\nPerplexity: ', lda.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    #
    # # Compute Coherence Score
    # coherence_model_lda = CoherenceModel(model=lda, texts=corpus, dictionary=dict, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # print('\nCoherence Score: ', coherence_lda)

    # Visualize the topics
    # pyLDAvis.enable_notebook()
    # vis = pyLDAvis.gensim.prepare(ldamallet, corpus, dict)
    # vis


def pick_most_important_phrases_using_tfidf(topn=10):
    sql_programming = "select ppl.stem from pracujpl ppl inner join job_titles jt on jt.pos_annotation='nazwa języka programowania' and ppl.offerData_jobTitle LIKE '%'||jt.job_title||'%' where ppl.stem is not null"
    sql_salles = "select ppl.stem from pracujpl ppl where offerData_jobTitle LIKE '%sprzedawca%' and ppl.stem is not null"
    sql_drivers = "select ppl.stem from pracujpl ppl where offerData_jobTitle LIKE '%kierowca międzynarodowy%' and ppl.stem is not null"
    engine = create_sqlite_engine()
    corpus = engine.execute(sql_programming).fetchall()
    logger.info("Data read. Corpus size: {}".format(len(corpus)))
    corpus = [txt[0].split() for txt in corpus]
    corpus = applyPhrases(corpus)
    logger.info("Phrases applied")
    stopwords = get_stopwords()
    corpus = [[word for word in doc if word not in stopwords] for doc in corpus]
    logger.info("Stopwords removed")
    dict = Dictionary(corpus)
    logger.info("Dict created")
    corpus_bow = [dict.doc2bow(text) for text in corpus]
    logger.info("BOW calculated")
    logger.info("TOP-{} using TFIDF".format(str(topn)))
    model = TfidfModel(corpus_bow, id2word=dict)  # fit model
    model = TfidfModel(corpus_bow, id2word=dict)  # fit model
    corpus_tfidf = model[corpus_bow]

    tfidf_data = [[dict[word] for word, tfids_score in sorted(doc_tfids, key=lambda tup: tup[1], reverse=True)[:topn]] for doc_tfids in corpus_tfidf]
    tfidf_data_dict = Dictionary(tfidf_data)
    top_words_set = list(set([top_word for doc_top in tfidf_data for top_word in doc_top]))
    top_word_df = [tfidf_data_dict.dfs[tfidf_data_dict.token2id[s]] for s in top_words_set]
    top_word_stats = zip(top_words_set, top_word_df)
    print("Most important words total:")
    print(sorted(top_word_stats, key=lambda tup: tup[1], reverse=True))
    print("Most important words per doc:")
    # for doc_tfids in corpus_tfidf:
        # print([dict[word] for word, tfids_score in sorted(doc_tfids, key=lambda tup: tup[1], reverse=True)[:topn]])


def detect_keywords_using_textrank_summarozation():
    sql_programming = "select ppl.stem from pracujpl ppl inner join job_titles jt on jt.pos_annotation='nazwa języka programowania' and ppl.offerData_jobTitle LIKE '%'||jt.job_title||'%' where ppl.stem is not null"
    sql_salles = "select ppl.stem from pracujpl ppl where offerData_jobTitle LIKE '%sprzedawca%' and ppl.stem is not null"
    sql_drivers = "select ppl.stem from pracujpl ppl where offerData_jobTitle LIKE '%kierowca międzynarodowy%' and ppl.stem is not null"
    engine = create_sqlite_engine()
    corpus = engine.execute(sql_salles).fetchall()
    logger.info("Data read. Corpus size: {}".format(len(corpus)))
    corpus = [txt[0].split() for txt in corpus]
    corpus = applyPhrases(corpus)
    logger.info("Phrases applied")
    stopwords = get_stopwords()
    corpus = [[word for word in doc if word not in stopwords] for doc in corpus]
    logger.info("Stopwords removed")
    logger.info("Detecting keywords")
    keyword_data = [keywords(' '.join(doc)).split('\n') for doc in corpus]

    keyword_data_dict = Dictionary(keyword_data)
    top_words_set = list(set([top_word for doc_top in keyword_data for top_word in doc_top]))
    top_word_df = [keyword_data_dict.dfs[keyword_data_dict.token2id[s]] for s in top_words_set]
    pos_data = [extract_pos(word.split()) for word in top_words_set]
    pos_elements, pos_annotations = zip(*pos_data)

    top_word_stats = zip(top_words_set, top_word_df, pos_elements)
    print("Most important words total:")
    top_word_stats = sorted(top_word_stats, key=lambda tup: tup[1], reverse=True)
    top_words_pd = pd.DataFrame(top_word_stats, columns=['keyword', 'df', 'pos'])
    top_word_groups = [', '.join(similar_words) for similar_words in top_words_pd.groupby('pos')['keyword'].apply(list)]
    for group in top_word_groups:
        print(group)



if __name__ == "__main__":
    # https://github.com/nikita-moor/morfeusz-wrapper
    result = m.analyse("Jira")
    logger.info("START")
    # move_csv_to_db()
    # preprocess_corpus_and_find_phrases_and_stats()
    # extract_main_job_titles()
    # perform_topic_modelling()
    # pick_most_important_phrases_using_tfidf(topn=40)
    detect_keywords_using_textrank_summarozation()
    logger.info("DONE")
