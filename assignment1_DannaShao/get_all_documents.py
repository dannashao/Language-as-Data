import http.client, urllib.parse, json
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
import html5lib
import os
import sys
from tqdm import tqdm

######## CREATE OUTPUT FOLDERS ########
def create_folders_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def create_folders(dir):
    '''
    Creates folders in the directory if they do not exist
    Folder Structure:
    ├── * input directory
    │   ├── eng
    │   │   ├── train
    │   │   ├── test
    │   ├── nld
    │   │   ├── train
    │   │   ├── test
    '''
    create_folders_if_not_exist(dir)
    
    lang1_dir = os.path.join(dir, r'eng')
    create_folders_if_not_exist(lang1_dir)
    lang1_train_dir = os.path.join(lang1_dir, r'train')
    create_folders_if_not_exist(lang1_train_dir)
    lang1_test_dir = os.path.join(lang1_dir, r'test')
    create_folders_if_not_exist(lang1_test_dir)
        
    lang2_dir = os.path.join(dir, r'nld')
    create_folders_if_not_exist(lang2_dir)
    lang2_train_dir = os.path.join(lang2_dir, r'train')
    create_folders_if_not_exist(lang2_train_dir)
    lang2_test_dir = os.path.join(lang2_dir, r'test')
    create_folders_if_not_exist(lang2_test_dir)

    return lang1_train_dir, lang1_test_dir, lang2_train_dir, lang2_test_dir


######## GUARDIAN SCRAPE ########
def guardian_query(keyword, from_date, to_date, page='1', page_size='200'):
    '''
    Get query content
    example usage: guardian_query(keyword='abortion',from_date='2022-06-24',to_date='2022-07-24')
    '''
    
    api_key = 'ca17bcab-db02-44a8-8a32-dd408dfbffcd'
    
    conn = http.client.HTTPConnection('content.guardianapis.com')
    params = ('from-date=' + str(from_date) + '&to-date=' + str(to_date) + 
              '&page=' + str(page) + '&page-size=' + str(page_size) + '&q=' + str(keyword) +
              '&api-key=' + str(api_key))
    
    conn.request('GET', '/search?{}'.format(params))

    res = conn.getresponse()
    data = res.read()
    query_content=(data.decode('utf-8'))
    query = json.loads(query_content)
    
    return query
 
#query_abortion_1 = guardian_query(keyword='abortion',from_date='2022-06-24',to_date='2022-09-24', page='1')
#query_abortion_2 = guardian_query(keyword='abortion',from_date='2022-06-24',to_date='2022-09-24', page='2')
#query_abortion_3 = guardian_query(keyword='abortion',from_date='2022-06-24',to_date='2022-09-24', page='3')
#query = [query_abortion_1,query_abortion_2,query_abortion_3]


def get_guardian_metadata(query):
    section = []
    date = []
    title = []
    urls = []
    for i in range(len(query)):
        articles = (query[i]['response'])['results']
        max_link = 1000
        for j, article in enumerate(articles):
            if j < max_link:
                section.append(article['sectionName'])
                date.append(article['webPublicationDate'])
                title.append(article['webTitle'])
                urls.append(article['webUrl'])
    return section, date, title, urls


#metadata_abortion = get_guardian_metadata(query)
#url_abortion = metadata_abortion[3]


def metadata_train_test_split(metadata, n = 20):
    '''
    Split the train/test with n% of test data
    '''
    df = pd.DataFrame(metadata).T.rename(columns={0:'section', 1:'time', 2:'title', 3:'url'})
    rdf = df.sample(frac=1, random_state=1) # reproducibility
    
    test = int(len(df)*(n/100))
    train = len(df)
    
    test_index = rdf.index[0:test]
    train_index = rdf.index[test:train]
    
    test_metadata = rdf.head(test)
    train_metadata = rdf.tail(train-test)
    
    return train_metadata, test_metadata, train_index, test_index

#splittedmetadata = metadata_train_test_split(metadata_abortion)


def write_guardian_metadata(outfile_keywords, splitted_metadata, dirs):
    trainfile = dirs[0] + '/' + outfile_keywords + "_train_metadata.csv"
    testfile = dirs[1] + '/' + outfile_keywords + "_test_metadata.csv"
    splittedmetadata[0].to_csv(trainfile)
    splittedmetadata[1].to_csv(testfile)


def get_guardian_content(url):
    '''
    Get the text from url link
    '''
    soup = BeautifulSoup(requests.get(url).text,"html5lib")
    text = ""
    for item in soup.select("p",{"class":"dcr-1kas69x"}): # Guardian body text class
        text += item.text.strip()
    return text


def write_guardian_output(splitted_metadata, outfile_keywords, dirs):
    # train
    train = splitted_metadata[0].reset_index()
    print("Scraping English training data from the Guardian..")
    for i in tqdm(range(len(train))):
        title = train['title'][i]
        url = train['url'][i]
        text = get_guardian_content(url)
        dir = dirs[0]
        filename = outfile_keywords + "_" + str(i) + "_" + str(title).replace('/','_') + ".txt"
        with open(dir + "/" + filename, "w", encoding = "utf-8") as f:
            f.write(text)
    
    # test
    test = splitted_metadata[1].reset_index()
    print("Scraping English test data from the Guardian..")
    for j in tqdm(range(len(test))):
        title = test['title'][j]
        url = test['url'][j]
        text = get_guardian_content(url)
        dir = dirs[1]
        filename = outfile_keywords + "_" + str(i) + "_" + str(title).replace('/','_') + ".txt"
        with open(dir + "/" + filename, "w", encoding = "utf-8") as f:
            f.write(text)


######## NOS SCRAPE ########
def url_to_html(url):
    """Scrapes the html content from a web page. Takes a URL string as input and returns an html object. """
    # THIS FUNCTION COMES FROM Lab 1 of the course Language as Data 2023, Vrije Universiteit Amsterdam. 
    # Get the html content
    headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
    }
    res = requests.get(url, headers=headers)
    #res = requests.get(url + ".pdf", headers={"User-Agent": "XY"})
    html = res.text
    parser_content = BeautifulSoup(html, 'html5lib')
    return parser_content


def get_nos_metadata(keyword, page_range):
    '''
    Scrape NOS metadata from the search page since it does not have an official API
    '''
    urls = []
    titles = []
    times = []
    domain = "https://nos.nl/"
    for i in range(1,page_range+1):
        search_url = "https://nos.nl/zoeken/?q=" + keyword + "&page=" + str(i)
        parser_content= url_to_html(search_url)
        result_url = parser_content.find_all("a", {"class":"sc-f75afcb6-4 isiLEZ"})
        result_times = parser_content.find_all("div",{"class":"sc-d6d7be46-0 jjSfnY sc-f75afcb6-6 hGGBnM"})
        result_titles = parser_content.find_all("h2",{"class":"sc-f75afcb6-3 lhteiV"})
        for j, url in enumerate(result_url):
            article_url = domain + url["href"]
            urls.append(article_url)
        for m, title in enumerate(result_titles):
            article_title = re.search(r'\<h2 class\=\"sc\-f75afcb6\-3 lhteiV\"\>(.*?)\<\/h2\>', str(title)).group(1)
            titles.append(article_title)
        for k,time in enumerate(result_times):
            article_time = re.search(r'time datetime\=\"(.*?)\"\>', str(time)).group(1)
            times.append(article_time)
    return times, titles, urls

nos_metadata = get_nos_metadata(keyword = "abortus", page_range = 35) # scraping some extra links as there may be invalid video links


def remove_nos_videos(metadata):
    '''
    NOS search results sometimes includes video links
    example: https://nos.nl/nieuwsuur/video/2496544-cu-lid-annelijn-door-wie-wil-je-eigenlijk-aardig-gevonden-worden
    This function removes such links
    '''
    df = pd.DataFrame(metadata).T.rename(columns={0:'time', 1:'title', 2:'url'})
    video_pattern = "\/video\/"
    filter = df['url'].str.contains(video_pattern)
    cleaned_df = df[~filter].head(600)
    
    return cleaned_df

cleaned_nos_metadata = remove_nos_videos(nos_metadata)


def remove_nos_videos(metadata):
    '''
    NOS search results sometimes includes video links
    example: https://nos.nl/nieuwsuur/video/2496544-cu-lid-annelijn-door-wie-wil-je-eigenlijk-aardig-gevonden-worden
    This function removes such links
    '''
    df = pd.DataFrame(metadata).T.rename(columns={0:'time', 1:'title', 2:'url'})
    video_pattern = "\/video\/"
    filter = df['url'].str.contains(video_pattern)
    cleaned_df = df[~filter].head(600)
    
    return cleaned_df


def nos_metadata_train_test_split(df, n = 20):
    '''
    Split the train/test with n% of test data for nos
    '''
    #df = pd.DataFrame(metadata).T.rename(columns={0:'time', 1:'title', 2:'url'})
    rdf = df.sample(frac=1, random_state=1) # reproducibility
    
    test = int(len(df)*(n/100))
    train = len(df)
    
    test_index = rdf.index[0:test]
    train_index = rdf.index[test:train]
    
    test_metadata = rdf.head(test)
    train_metadata = rdf.tail(train-test)
    
    return train_metadata, test_metadata, train_index, test_index

nos_splittedmedatada = nos_metadata_train_test_split(cleaned_nos_metadata)


def get_nos_content_and_section(url):
    '''
    Get the text and section from url link
    This is because of the section is not stored in the search page but the main article
    '''
    #print(url)
    soup = BeautifulSoup(requests.get(url).text,"html5lib")
    section_container = soup.select("p",{"class":"sc-f9df6382-7 cMuisv"}) # NOS section class
    section = re.search(r'cMuisv\"\>(.*?)\<\/p\>', str(section_container)).group(1)
    text = ""
    for item in soup.select("p",{"class":"sc-6d77a1d1-0 chzewu"}): # NOS body text class
        text += item.text.strip()
    return text, section


def write_nos_output_and_section(splitted_metadata, keyword, dirs):
    '''
    This function not only writes the output files but also store the section information
    since the section information is scraped from the article content
    '''
    # train
    print("Scraping Dutch training data from NOS..")
    train_section = []
    train = splitted_metadata[0].reset_index()
    for i in tqdm(range(len(train))):
        title = train['title'][i]
        url = train['url'][i]
        nos = get_nos_content_and_section(url)
        section = nos[1]
        train_section.append(section)
        text = nos[0]
        dir = dirs[2]
        filename = keyword + "_" + str(i) + "_"+ str(title).replace('/','_') + ".txt"
        with open(dir + "/" + filename, "w", encoding = "utf-8") as f:
            f.write(text)
    
    # test
    print("Scraping Dutch test data from NOS..")
    test_section = []
    test = splitted_metadata[1].reset_index()
    for j in tqdm(range(len(test))):
        title = test['title'][j]
        url = test['url'][j]
        nos = get_nos_content_and_section(url)
        section = nos[1]
        test_section.append(section)
        text = nos[0]
        dir = dirs[3]
        filename = keyword + "_" + str(j) + "_" + str(title).replace('/','_') + ".txt"
        with open(dir + "/" + filename, "w", encoding = "utf-8") as f:
            f.write(text)

    return train_section, test_section    


def write_nos_metadata(dirs, outfile_keywords, splittedmetadata, sections):
    '''
    This function combines the content to metadata and writes the output csv files
    '''
    trainfile = dirs[2] + '/' + outfile_keywords + "_train_metadata.csv"
    testfile = dirs[3] + '/' + outfile_keywords + "_test_metadata.csv"
    # add the content column to the metadata dataframe
    splittedmetadata[0]['content'] = sections[0]
    splittedmetadata[0]['content'] = sections[0]
    splittedmetadata[1]['content'] = sections[1]
    # save the dataframe to csv
    splittedmetadata[0].to_csv(trainfile)
    splittedmetadata[1].to_csv(testfile)




def main(argv=None):
    if argv is None:
        argv = sys.argv

    out_dir = argv[1]
    dirs = create_folders(out_dir)
    #write_guardian_metadata(outfile_keywords = 'abortion', splitted_metadata = splittedmetadata, dirs=dirs)
    #write_guardian_output(outfile_keywords = 'abortion', splitted_metadata = splittedmetadata, dirs=dirs)

    sections = write_nos_output_and_section(splitted_metadata = nos_splittedmedatada, keyword = 'abortus', dirs=dirs)
    write_nos_metadata(dirs = dirs, outfile_keywords='abortus', splittedmetadata=nos_splittedmedatada, sections=sections)
    
if __name__ == '__main__':
    main()
