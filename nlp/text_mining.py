import os
import re
import gensim
import pickle
import time
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import LdaModel
from gensim.models import Word2Vec
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
import numpy as np
from sklearn.manifold import TSNE

# !/usr/bin/env python -W ignore::DeprecationWarning

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
            "SELECT id, content FROM pracujpl WHERE content is not null and lang='pl'").fetchall()
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
                        txt = [word if word not in error_phrase_replacements else error_phrase_replacements[word] for
                               word
                               in txt.split()]
                        stem = [extract_stem(m.analyse(word), m, error_phrase_replacements) for word in txt]
                        txt = ' '.join(txt)
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


def detect_and_save_phrases():
    engine = create_sqlite_engine()
    data = engine.execute(
        "SELECT id, stem FROM pracujpl WHERE stem is not null and lang='pl'").fetchall()
    logger.info("Data read")
    logger.info("Transforming texts start")
    ids, corpus = zip(*data)
    corpus = [doc.split() for doc in corpus]
    logger.info("Transforming texts DONE")
    logger.info('Applying phrases')
    phrases = applyPhrases(corpus)
    logger.info('Phrases applied')
    update_data = zip(ids, phrases)
    for id, phrase in update_data:
        engine.execute("UPDATE pracujpl SET stem_phrases='{0}' where id={1}".format(' '.join(phrase), str(id)))


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

    tfidf_data = [[dict[word] for word, tfids_score in sorted(doc_tfids, key=lambda tup: tup[1], reverse=True)[:topn]]
                  for doc_tfids in corpus_tfidf]
    tfidf_data_dict = Dictionary(tfidf_data)
    top_words_set = list(set([top_word for doc_top in tfidf_data for top_word in doc_top]))
    top_word_df = [tfidf_data_dict.dfs[tfidf_data_dict.token2id[s]] for s in top_words_set]
    top_word_stats = zip(top_words_set, top_word_df)
    print("Most important words total:")
    print(sorted(top_word_stats, key=lambda tup: tup[1], reverse=True))
    print("Most important words per doc:")
    # for doc_tfids in corpus_tfidf:
    # print([dict[word] for word, tfids_score in sorted(doc_tfids, key=lambda tup: tup[1], reverse=True)[:topn]])


def detect_keywords(explicit_sql=None, job_category='All'):
    sql_programming = "select ppl.stem_phrases from pracujpl ppl inner join job_titles jt on jt.pos_annotation='nazwa języka programowania' and ppl.offerData_jobTitle LIKE '%'||jt.job_title||'%' where ppl.stem_phrases is not null"
    sql_salles = "select ppl.stem_phrases from pracujpl ppl where offerData_jobTitle LIKE '%sprzedawca%' and ppl.stem_phrases is not null"
    sql_sales_representative = "select ppl.stem_phrases from pracujpl ppl where (offerData_jobTitle like '%handlowy%' or offerData_jobTitle like '%handlowiec%') and ppl.stem_phrases is not null"
    sql_drivers = "select ppl.stem_phrases from pracujpl ppl where offerData_jobTitle LIKE '%kierowca międzynarodowy%' and ppl.stem_phrases is not null"
    engine = create_sqlite_engine()
    sql = sql_programming
    if explicit_sql is not None:
        sql = explicit_sql
    logger.info(sql)
    corpus = engine.execute(sql).fetchall()
    logger.info("Data read. Corpus size: {}".format(len(corpus)))
    corpus = [txt[0].split() for txt in corpus]
    stopwords = get_stopwords()
    corpus = [[word for word in doc if word not in stopwords] for doc in corpus]
    logger.info("Stopwords removed")
    logger.info("Detecting keywords")
    logger.info("---TextRank")
    keyword_data_textrank = extract_keywords(corpus, mode='textrank')
    logger.info("---TFiDF")
    keyword_data_tfidf = extract_keywords(corpus, mode='tfidf', tfidf_topn=20)
    word_sets = []
    top_words = set()

    for keyword_data, name in [(keyword_data_textrank, 'TextRank'), (keyword_data_tfidf, 'TFiDF')]:
        logger.info("Calculating stats [{}]".format(name))
        keyword_data_dict = Dictionary(keyword_data)
        top_words_set = list(set([top_word for doc_top in keyword_data for top_word in doc_top]))
        if len(word_sets) > 0:
            logger.info("Removing duplicates [{}]".format(name))
            top_words_set = [word for word in top_words_set if word not in word_sets[len(word_sets) - 1]]
            logger.info("After duplicates removal: {0}, [{1}]".format(len(top_words_set), name))
            if len(top_words_set) == 0:
                logger.info("Keyword dictionary is empty, skipping [{}]".format(name))
                continue
        word_sets.append(top_words_set)
        top_word_df = [keyword_data_dict.dfs[keyword_data_dict.token2id[s]] for s in top_words_set]
        # logger.info("Adding POS data [{}]".format(name))
        # pos_data = [extract_pos(word.split()) for word in top_words_set]
        # pos_elements, pos_annotations = zip(*pos_data)

        top_word_stats = zip(top_words_set, top_word_df)  # , pos_elements)
        top_word_stats = [word for word, df in sorted(top_word_stats, key=lambda tup: tup[1], reverse=True)
                          if df >= int(len(corpus) / 100)]  # 300 - 350
        top_words.update(top_word_stats)
        # top_words_pd = pd.DataFrame(top_word_stats, columns=['keyword', 'df', 'pos'])
        # top_word_groups = [', '.join(similar_words) for similar_words in top_words_pd.groupby('pos')['keyword'].apply(list)]
        # logger.info("Most important words [{}]:".format(name))
        # for group in top_word_groups:
        #     print(group)
        # word2vec_plot([word for word in word_sets[0] if ' ' not in word])

    # common_words = [word for word in word_sets[0] if word in word_sets[1]]
    # logger.info("Common words:")
    # print(common_words)
    print(group_keywords_by_examples(top_words))


def extract_keywords(corpus, mode='textrank', tfidf_topn=20):
    data = []
    if mode == 'textrank':
        data = [keywords(' '.join(doc)).split('\n') for doc in corpus]
    elif mode == 'tfidf':
        dict = Dictionary(corpus)
        corpus_bow = [dict.doc2bow(text) for text in corpus]
        model = TfidfModel(corpus_bow, id2word=dict)  # fit model
        corpus_tfidf = model[corpus_bow]
        data = [
            [dict[word] for word, tfids_score in sorted(doc_tfids, key=lambda tup: tup[1], reverse=True)[:tfidf_topn]]
            for doc_tfids in corpus_tfidf]
    return data


def calculate_word_embeddings():
    sql = "select stem from pracujpl where stem is not null and lang='pl'"
    vector_size = 200
    engine = create_sqlite_engine()
    corpus = engine.execute(sql).fetchall()
    logger.info("Data read. Corpus size: {}".format(len(corpus)))
    corpus = [txt[0].split() for txt in corpus]
    corpus = applyPhrases(corpus)
    logger.info("Phrases applied")
    logger.info("Word2Vec rock'n'roll start!")
    model = Word2Vec(corpus, size=vector_size, window=5, min_count=1, workers=6)
    model.save("word2vec_all_200_5_fixed.model")
    logger.info("Word2Vec saved")
    word_vectors = model.wv
    del model
    # logger.info("Building global dictionary")
    # corpus_dict = Dictionary(corpus)
    # embeddings = []
    # logger.info("Building word embedding dictionary")
    # for word in corpus_dict.token2id.keys():
    #     embeddings.append((word, word_vectors[word]))
    # logger.info("Saving data to DB")
    # embeddings_pd = pd.DataFrame(embeddings, columns=['word', 'vector_{}'.format(str(vector_size))])
    # embeddings_pd.to_sql('doc2vec_{}'.format(str(vector_size)), con=engine, if_exists='replace', index_label='id')
    logger.info("Word embeddings dict saved")


def group_keywords_by_examples(top_words):
    model = Word2Vec.load("word2vec_all_200_5_fixed.model")

    similarity_threshold = 0.40
    top_word_groups = {}
    examples = {
        'skills soft': [
            'Pokrzepiający', 'Inspirator', 'Rozmowny', 'Dynamiczny', 'Przedsiębiorczy', 'Przekonujący', 'Zaradny',
            'Pewny', 'Autorytatywny', 'Pewny_siebie', 'Niezależny', 'Zdecydowany', 'Działacz', 'Wytrwały', 'Lider',
            'Produktywny ', 'Analityczny', 'Rozważny', 'Zorganizowany', 'Uporządkowany', 'Drobiazgowy', 'Kulturalny',
            'Elastyczny', 'Opanowany', 'Powściągliwy', 'Cierpliwy', 'Życzliwy', 'Dyplomatyczny', 'Konsekwetny',
            'Taktowny', 'Mediator', 'Tolerancyjny', 'Słuchacz', 'Zrównoważony', 'Asertywny', 'Samodzielny',
            'Punktualny', 'Rozwiązywać problem', 'Efektywny', 'Odporny_na_stres', 'odpowiedzialny', 'samodzielny',
            'profesjonalny', 'komunikatywny', 'przemysłowy', 'odpowiedzialnosc', 'samodzielnosc', 'gotowość do praca',
            'dyspozycyjność', 'osiągać wynik', 'kreatywność', 'rzetelność', 'estetyczny wygląd', 'sumienność',
            'wytrwałość', 'aktywny', 'analityczny', 'pasja', 'inicjatywa', 'energia', 'przedsiębiorczy', 'zgrany',
            'interpersonalny', 'systematyczny', 'zespołowy', 'pozytywny', 'zdeterminowany',
            'łatwość nawiązywać kontakt', 'silny motywacja', 'szybki uczenie się', 'podnoszenie swój kwalifikacja',
            'zarządzanie ryzyko', 'poczucie humor', 'samodyscyplina', 'dociekliwość', 'managerski', 'przywódczy',
            'charyzma', 'nadzorować', 'zaangażowanie', 'ścisły', 'humanistyczny', 'artystyczny', 'dbałość',
            'skrupulatność', 'uczciwość', 'podejmować decyzja', 'poszukiwać', 'odporność', 'otwartość', 'radzić soba',
            'poczucie estetyka', 'negocjacje', 'operatywność', 'asertywność', 'lekkość', 'łatwość operować słowo'
        ],
        'skills_hard': [
            'Jira', 'Agile', 'Scrum', 'PMP', 'wykształcenie',
            'Power_Point', 'Excel', 'Word', 'BPMN', 'UML', 'Confluence', 'ITIL', 'Visio', 'PMI', 'marketing', 'audyt',
            'techniczny', 'informatyczny', 'angielski', 'niemiecki', 'elektryczny', 'produktowy', 'telefoniczny',
            'kadrowy', 'telekomunikacyjny', 'kredytowy', 'detaliczny', 'ekonomiczny', 'prawo_jazda_kat_bit',
            'rachunkowość', 'właściwy sposób ekspozycja towar', 'znajomość rynek', 'znajomość pakiet',
            'wykształcenie wysoki', 'kilkuletni', 'prawniczy', 'urzędniczy', 'administracyny', 'budowlany', 'finansowy',
            'farmaceutyczny', 'sprzedażowy', 'medyczny', 'bankowy', 'telefoniczny', 'elektryczny', 'biegły',
            'telekomunikacyjny', 'ubezpieczeniowy', 'logistyczny', 'kredytowy', 'transportowy', 'biurowy', 'kadrowy',
            'elektroniczny', 'wdrożeniowy', 'sieciowy', 'serwisowy', 'chemiczny', 'elektrotechniczny',
            'kosmetyczny', 'menedżerski', 'kurierski', 'niemieckojęzyczny', 'graficzny', 'windykacyjny', 'materiałowy',
            'konstrukcyjny', 'hydrauliczny', 'meblowy', 'statystyczny', 'poligraficzny', 'minimum letni', 'metodyka',
            'scrum', 'kanban', 'autocad', 'udokumentować', 'projektować', 'projekt graficzny', 'Photoshop',
            'corel draw', 'ulotka', 'montaż', 'poligraficzny', 'adobe', 'fotografia', 'power point', 'ms office',
            'reklama', 'agencja reklamowy', 'ilustrator', 'illustrator', 'ilustracja', 'design', 'DTP', 'folder',
            'skład', 'dziennikarz', 'internet', 'programować', 'webowy', 'JavaScript', 'materiał marketingowy',
            'pisanie', 'media', 'redakcja', 'HTML', 'redagować tekst', 'treści', 'przewóz', 'miedzynarodowy',
            'przewóz rzecz', 'przewóz ludzie', 'koordynować', 'prawo jazda'
        ],
        'benefits': [
            'benefit', 'innowacyjny', 'atrakcyjny wynagrodzenie', 'stabilny warunki', 'umowa praca',
            'samochód służbowy', 'dofinansować', 'własny działalność gospodarczy', 'własny działalność gospodarczy',
            'prywatny opieka medyczny', 'pakiet ms office', 'wyprawka', 'dofinansować', 'długoterminowy', 'długofalowy',
            'prowizyjny', 'bonusowy', 'rekreacyjny', 'system premiowy', 'premia uzależniony oda', 'miły atmosfera',
            'ubezpieczenie grupowy', 'samochód laptop telefon', 'częsty podróż służbowy', 'różny pora dzień', 'b2b',
            'multisport', 'pakiet socjalny', 'pełny wymiar czas', 'częsty wyjazd', 'awans'
        ]}
    multi_word_top_words = [word for word in top_words if ' ' in word]
    for group in examples.keys():
        if group not in top_word_groups:
            top_word_groups[group] = set()
        for keyword in examples[group]:
            keyword = keyword.lower()
            sims = []
            keyword_vector = None
            if ' ' in keyword:
                try:
                    keyword_vector = sum(model[keyword.split()])
                    sims = model.similar_by_vector(keyword_vector, topn=30)
                    print("Multiword example case:")
                    print(keyword)
                    print(sims)
                except KeyError:
                    logger.warning("Worn in example phrase: '{}' not in dictionary!!!".format(keyword))
                    continue
            else:
                try:
                    keyword_vector = model[keyword]
                    sims = model.wv.similar_by_word(keyword, topn=30)
                except KeyError:
                    logger.warning("'{}' not in dictionary!!!".format(keyword))
                    continue
            multiword_sims = []
            if len(multi_word_top_words) > 0 and keyword_vector is not None:
                for phrase in multi_word_top_words:
                    try:
                        if cosine(keyword_vector, sum(model[phrase.split()])) >= similarity_threshold:
                            multiword_sims.append(phrase)
                    except KeyError:
                        logger.warning("Word in phrase '{}' not in dictionary!!!".format(phrase))
                        continue
                if len(multiword_sims) > 0:
                    print("Multiword keywords case:")
                    print(keyword)
                    print(multiword_sims)
            sims = [sim[0] for sim in sims if sim[0] in top_words and sim[1] >= similarity_threshold]
            sims.extend(multiword_sims)
            if keyword in top_words:
                sims.append(keyword)
            if len(sims) > 0:
                top_word_groups[group].update(sims)
    logger.info("Original set of top words:")
    print(top_words)
    return top_word_groups


def word2vec_plot(vocab):
    model = Word2Vec.load("word2vec_all_200_5_fixed.model")
    logger.info("Creating word2vec dict")
    valid_vocab = []
    for word in vocab:
        try:
            model[word]
            valid_vocab.append(word)
        except KeyError:
            pass
    X = model[valid_vocab]
    logger.info("T-SNE start")
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    logger.info("T-SNE done")
    df = pd.DataFrame(X_tsne, index=valid_vocab, columns=['x', 'y'])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(df['x'], df['y'])
    for word, pos in df.iterrows():
        ax.annotate(word, pos)
    plt.show()


def cosine(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


if __name__ == "__main__":
    # https://github.com/nikita-moor/morfeusz-wrapper
    logger.info("START")

    # model = Word2Vec.load("word2vec_all_200_5_fixed.model")
    # keyword = 'gotowość do praca'
    # sentence_vector_1 = sum(model[keyword.split()])
    # keyword = 'gotować'
    # sentence_vector_2 = sum(model[keyword.split()])
    # print(cosine(sentence_vector_1, sentence_vector_2))

    # print(model.similar_by_vector(sentence_vector_1, topn=30))
    # move_csv_to_db()
    # preprocess_corpus_and_find_phrases_and_stats()
    # extract_main_job_titles()
    # perform_topic_modelling()
    # pick_most_important_phrases_using_tfidf(topn=40)

    detect_keywords(
        explicit_sql="select ppl.stem_phrases from pracujpl ppl where offerData_jobTitle LIKE '%Przedstawiciel Handlowy%' and ppl.stem_phrases is not null",
        job_category='IT')
    # TODO: remove polish chars to catch: 'przywodczy'
    # TODO: examples categories ('All', 'IT', 'Business', 'Accounting', 'Physical', 'Salles', 'Marketing/Advertising', 'Design', 'Manager')

    # calculate_word_embeddings()
    # word2vec_testing()
    logger.info("DONE")
