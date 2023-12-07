import pandas as pd
import string
import statistics
import numpy as np
from scipy.stats import spearmanr

import os
import sys
import click
from tqdm.auto import tqdm

from keybert import KeyBERT
from wordcloud import WordCloud
from textblob import TextBlob
from textblob_nl import PatternAnalyzer
import textstat

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels


############ USING STANZA CHECK ############
use_stanza = False
use_pickle = False
use_kw = False
print("Please put this script under the same folder as eng and nld data.")
print("***IMPORTANT INFORMATION***")
print("stanza model takes long to load and it may take extremely long to process. You can select to use pre-processed texts.")
if click.confirm('Do you want to use stanza? [SLOW]', default=False):
    use_stanza = True
if click.confirm('Do you want to use stanza pre-processed articles?', default=True):
    use_pickle = True
if use_stanza == True:
    import stanza
if use_pickle == True:
    import pickle
    use_stanza = False
if click.confirm('Keyword calculation uses around 5-10 minutes. Do you want to use pre-calculated keywords?', default=False):
    use_kw = True
    kw = pd.read_csv('keys.csv')
print("Using stanza: ", use_stanza)
print("Using pre-processed articles: ", use_pickle)
print("Using pre-calculated keywords: ", use_kw)
print("+++++++++ STARTING ANALYZE +++++++++")


############ DATA PATHS ############
eng_train_meta = pd.read_csv('eng/train/abortion_train_metadata.csv')
eng_train_path = 'eng/train/'
eng_test_meta = pd.read_csv('eng/test/abortion_test_metadata.csv')
eng_test_path = 'eng/test/'

nld_train_meta = pd.read_csv('nld/train/abortus_train_metadata.csv').rename(columns={"content": "section"})
nld_train_path = 'nld/train/'
nld_test_meta = pd.read_csv('nld/test/abortus_test_metadata.csv').rename(columns={"content": "section"})
nld_test_path = 'nld/test/'

save_results_to = 'results/'
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

def get_all_article(path):
    import os,glob
    folder_path = path
    articles = []
    for filename in glob.glob(os.path.join(folder_path, '*.txt')):
        with open(filename, 'r') as f:
            text = f.read()
            articles.append(text)
    return articles

eng_train_articles = get_all_article(eng_train_path)
eng_test_articles = get_all_article(eng_test_path)
nld_train_articles = get_all_article(nld_train_path)
nld_test_articles = get_all_article(nld_test_path)


############ GENERAL STATISTICS ############
def avg_article_len(path):
    import os,glob
    folder_path = path
    filenum = 0
    wordcount = 0
    for filename in glob.glob(os.path.join(folder_path, '*.txt')):
        with open(filename, 'r') as f:
            text = f.read()
            filenum += 1
            wordcount += len(text)
    return wordcount/filenum

print("Average article length of ENG train: ", avg_article_len(eng_train_path))
print("Average article length of ENG test: ", avg_article_len(eng_test_path))
print("Average article length of NLD train: ", avg_article_len(nld_train_path))
print("Average article length of NLD test: ", avg_article_len(nld_test_path))

print("Average title length of ENG train: ", eng_train_meta['title'].apply(len).mean())
print("Average title length of ENG test: ", eng_test_meta['title'].apply(len).mean())
print("Average title length of NLD train: ", nld_train_meta['title'].apply(len).mean())
print("Average title length of NLD test: ", nld_test_meta['title'].apply(len).mean())

print("Initializing further calculations..")

def get_stylistic(articles):
    from collections import Counter
    ttr = [] # type-token ratio
    avg_sentence_len = []
    for article in articles:
        
        # Calculate TTR
        token_frequencies = Counter()
        for sentence in article.sentences:
            all_tokens =[token.text for token in sentence.tokens]
            token_frequencies.update(all_tokens)
        num_types = len(token_frequencies.keys())
        num_tokens = sum(token_frequencies.values())
        tt_ratio = num_types/float(num_tokens)
        ttr.append(tt_ratio)
        
        # Calculate average sentence length
        sentence_lengths =[len(sentence.tokens) for sentence in article.sentences]
        avg_sentence_len.append(statistics.mean(sentence_lengths))
    return ttr, avg_sentence_len

if ((use_stanza == False) & (use_pickle == True)):
    with open('processed_eng_train.pkl', 'rb') as f:
        processed_eng_train = pickle.load(f)
    with open('processed_nld_train.pkl', 'rb') as f:
        processed_nld_train = pickle.load(f)

def process_articles(articles, language):
    nlp = stanza.Pipeline(lang=language, processors='tokenize,pos,lemma')
    processed_articles = []
    print("Using stanza to process articles (it may take long...)")
    for article in tqdm(articles):
        processed_articles.append(nlp.process(article))
    return processed_articles

if ((use_stanza == True) & (use_pickle == False)):
    processed_eng_train = process_articles(eng_train_articles, language='en')
    processed_nld_train = process_articles(nld_train_articles,language='nl')


def print_stylistic():
    print("Average ttr of ENG train: ",statistics.mean(eng_train_stylistic[0]))
    print("Average sentence length of ENG train: ",statistics.mean(eng_train_stylistic[1]))
    print("Average ttr of NLD train: ",statistics.mean(nld_train_stylistic[0]))
    print("Average sentence length of NLD train: ",statistics.mean(nld_train_stylistic[1]))
    
if ((use_stanza == True)):
    eng_train_stylistic = get_stylistic(processed_eng_train)
    nld_train_stylistic = get_stylistic(processed_nld_train)
    print_stylistic()


def extract_keyword(articles, language):
    eng_kw_model = KeyBERT()
    nld_kw_model = KeyBERT(model = 'distiluse-base-multilingual-cased-v1')
    kw = []

    if language == 'eng':
        print("Calculating eng keywords")
        for article in tqdm(articles):
            keyword = eng_kw_model.extract_keywords(article)[0][0]
            kw.append(keyword)

    if language == 'nld':
        print("Calculating nld keywords")
        for article in tqdm(articles):
            keyword = nld_kw_model.extract_keywords(article)[0][0]
            kw.append(keyword)
    return kw



############ SENTIMENT ANALYSIS ############
def get_sentiment(articles, language):
    polarity = []
    subjectivity = []
    readability = []
    
    if language == 'eng':
        for article in articles:
            polarity.append(TextBlob(article).sentiment.polarity)
            subjectivity.append(TextBlob(article).sentiment.subjectivity)
            readability.append(textstat.flesch_reading_ease(article))

    if language == 'nld':
        for article in articles:
            polarity.append(TextBlob(text=article, analyzer=PatternAnalyzer()).sentiment[0])
            subjectivity.append(TextBlob(text=article, analyzer=PatternAnalyzer()).sentiment[1])
            readability.append(textstat.flesch_reading_ease(article))

    return polarity, subjectivity, readability


def create_df(meta, articles, language):
    df = pd.DataFrame(articles, columns=['content'])
    df['section'] = meta['section']
    df['keyword']= extract_keyword(articles, language)
    polarity, subjectivity, readability = get_sentiment(articles, language)
    df['polarity'] = polarity
    df['subjectivity'] = subjectivity
    df['readability'] = readability
    return df

if use_kw == False:
    eng_train_df = create_df(eng_train_meta, eng_train_articles, 'eng')
    eng_test_df = create_df(eng_test_meta, eng_test_articles, 'eng')
    nld_train_df = create_df(nld_train_meta, nld_train_articles, 'nld')
    nld_test_df = create_df(nld_test_meta, nld_test_articles, 'nld')

def create_df_with_kw(meta, articles, language, kwn):
    df = pd.DataFrame(articles, columns=['content'])
    df['section'] = meta['section']
    df['keyword'] = kw[kwn]
    polarity, subjectivity, readability = get_sentiment(articles, language)
    df['polarity'] = polarity
    df['subjectivity'] = subjectivity
    df['readability'] = readability
    return df

if use_kw == True:
    eng_train_df = create_df_with_kw(eng_train_meta, eng_train_articles, 'eng','eng_train')
    eng_test_df = create_df_with_kw(eng_test_meta, eng_test_articles, 'eng','eng_test')
    nld_train_df = create_df_with_kw(nld_train_meta, nld_train_articles, 'nld','nld_train')
    nld_test_df = create_df_with_kw(nld_test_meta, nld_test_articles, 'nld','nld_test')


############ COUNTRY ASSUMPTIONS ############
def separate_nld_country(nld_df):
    nld_country = nld_df.drop(columns=['content'])
    nld_country['section'] = np.where(nld_country['section'] == 'Buitenland', 'Foreign', nld_country['section'])
    nld_country['section'] = np.where(nld_country['section'] == 'Van Trump naar Biden', 'Foreign', nld_country['section'])
    nld_country['section'] = np.where(nld_country['section'] != 'Foreign', 'NL', nld_country['section'])
    
    print("+++ Domestic statistics +++")
    print(nld_country.loc[nld_country['section'] == 'NL'].mean())
    print("+++ Foreign statistics +++")
    print(nld_country.loc[nld_country['section'] == 'Foreign'].mean())
    return nld_country

def separate_eng_country(eng_df):
    eng_country = eng_df.drop(columns=['content'])
    eng_country['section'] = np.where(eng_country['section'] == 'US news', 'US', eng_country['section'])
    eng_country['section'] = np.where(eng_country['section'] != 'US', 'non_US', eng_country['section'])
    
    print("+++ US statistics +++")
    print(eng_country.loc[eng_country['section'] == 'US'].mean())
    print("+++ non_US statistics +++")
    print(eng_country.loc[eng_country['section'] == 'non_US'].mean())
    return eng_country

print("====== eng train statistics ======")
eng_train_country = separate_eng_country(eng_train_df)
print("====== eng test statistics ======")
eng_test_country = separate_eng_country(eng_test_df)

print("====== nld train statistics ======")
nld_train_country = separate_nld_country(nld_train_df)
print("====== nld test statistics ======")
nld_test_country = separate_nld_country(nld_test_df)

print("++++++++ Spearman correlations ++++++++")
print("++++++++ (country - polarity) ++++++++")
print("ENG train Spearman correlation (country - polarity):\n", spearmanr(eng_train_country['section'].replace({'US':-1, 'non_US':1}), eng_train_country['polarity']))
print("ENG test Spearman correlation (country - polarity):\n", spearmanr(eng_test_country['section'].replace({'US':-1, 'non_US':1}), eng_test_country['polarity']))
print("NLD train Spearman correlation (country - polarity):\n", spearmanr(nld_train_country['section'].replace({'NL':-1, 'Foreign':1}), nld_train_country['polarity']))
print("NLD test Spearman correlation (country - polarity):\n", spearmanr(nld_test_country['section'].replace({'NL':-1, 'Foreign':1}), nld_test_country['polarity']))

print("++++++++ (subjectivity - ABSpolarity) ++++++++")
print("ENG test Spearman correlation (subjectivity - ABSpolarity):\n", spearmanr(eng_test_country['subjectivity'], abs(eng_test_country['polarity'])))
print("NLD test Spearman correlation (subjectivity - ABSpolarity):\n", spearmanr(nld_test_country['subjectivity'], abs(nld_test_country['polarity'])))
print("ENG test Spearman correlation (subjectivity - readability):\n", spearmanr(eng_test_country['subjectivity'], eng_test_country['readability']))
print("NLD test Spearman correlation (subjectivity - readability):\n", spearmanr(nld_test_country['subjectivity'], nld_test_country['readability']))


############ PLOTTING ############
def create_wordcloud(df, figname):
    keywords = df['keyword'].to_string(index=False, header=False).replace(" ","").replace("\n"," ")
    wc = WordCloud().generate(keywords)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(save_results_to + figname + '.png', dpi = 100)
    plt.close()

create_wordcloud(eng_train_df, 'eng_keys')
create_wordcloud(nld_train_df, 'nld_keys')


def nld_section_and_pie(nld_df, figname, type='train'):
    nld_majors = ['Buitenland','Binnenland','Politiek']
    nld_sections = nld_df.drop(columns=['content'])
    nld_sections.loc[( (nld_sections['section'] != 'Buitenland') & (nld_sections['section'] != 'Binnenland') & (nld_sections['section'] != 'Politiek')), 'section'] = 'Others'
    nld_label_count = dict(nld_sections['section'].value_counts())
    nld_labels = list(nld_label_count.keys())
    nld_values = list(nld_label_count.values())
    plt.pie(nld_values,labels=nld_labels, autopct=lambda p : '{:.2f}% '.format(p,p * sum(nld_values)/100))
    if type == 'train':
        plt.title("nld train set sections")
    if type == 'test':
        plt.title("nld test set sections")
    #plt.show()
    plt.savefig(save_results_to + figname + '.png', dpi = 100)
    plt.close()
    return nld_sections

def eng_section_and_pie(eng_df, figname, type='train'):
    eng_sections = eng_df.drop(columns=['content'])
    eng_sections.loc[( (eng_sections['section'] != 'US news') & (eng_sections['section'] != 'Opinion') & (eng_sections['section'] != 'World news') & (eng_sections['section'] != 'Australia news')), 'section'] = 'Others'
    eng_label_count = dict(eng_sections['section'].value_counts())
    eng_labels = list(eng_label_count.keys())
    eng_values = list(eng_label_count.values())
    plt.pie(eng_values,labels=eng_labels, autopct=lambda p : '{:.2f}% '.format(p,p * sum(eng_values)/100))
    if type == 'train':
        plt.title("eng train set sections")
    if type == 'test':
        plt.title("eng test set sections")
    #plt.show()
    plt.savefig(save_results_to + figname + '.png', dpi = 100)
    plt.close()
    return eng_sections

print("plotting pie charts...")
eng_train_sections = eng_section_and_pie(eng_train_df,'eng_train_sections')
eng_test_sections = eng_section_and_pie(eng_test_df,'eng_test_sections')
nld_train_sections = nld_section_and_pie(nld_train_df,'nld_train_sections')
nld_test_sections = nld_section_and_pie(nld_test_df,'nld_test_sections')
print("section pie charts saved to /results")


# https://stackoverflow.com/a/50690729
def corrdot(*args, **kwargs): 
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)

def create_coorplot(df, figname):
    sns.set(style='white', font_scale=1.6)
    g = sns.PairGrid(df.drop(columns=['section']), aspect=1.4, diag_sharey=False)
    g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
    g.map_diag(sns.histplot, kde_kws={'color': 'black'})
    g.map_upper(corrdot)
    g.savefig(save_results_to + figname + '.png', dpi = 100)
    plt.close()

print("plotting correlation plots...")
create_coorplot(eng_train_df, 'eng_train_coor')
create_coorplot(eng_test_df, 'eng_test_coor')
create_coorplot(nld_train_df, 'nld_train_coor')
create_coorplot(nld_test_df, 'nld_test_coor')
print("correlation plots saved to /results")


def create_stripplot(section_df, figname):
    s = sns.stripplot(data=section_df, x='section', y='polarity',hue='subjectivity', legend = False)
    _, xlabels = plt.xticks()
    s.set_xticklabels(xlabels, size=10)
    #plt.show()
    plt.savefig(save_results_to + figname + '.png', dpi = 100)
    plt.close()
    
import warnings
warnings.filterwarnings("ignore")
create_stripplot(eng_train_sections, 'eng_train_strip')
create_stripplot(nld_train_sections, 'nld_train_strip')
create_stripplot(eng_test_sections, 'eng_test_strip')
create_stripplot(nld_test_sections, 'nld_test_strip')

def main(argv=None):
    if argv is None:
        argv = sys.argv

if __name__ == '__main__':
    main()