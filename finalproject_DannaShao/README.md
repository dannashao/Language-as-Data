# Code for Comparative Study of English and Dutch Abortion-themed Press Release Texts
Author: Danna Shao (2663369), d.shao@student.vu.nl

## Scraping Data and Building Corpus: get_all_documents.py
useage: python get_all_documents.py \<output_directory\>

### PURPOSE OF THE CORPUS: 
Use emotional analysis to discover the attitude difference between English and Dutch news on the topic abortion.

### EXPLANATION OF THE RETRIEVAL PROCEDURE:
- The English data is scraped from the guardians with its official api (https://open-platform.theguardian.com/documentation/).
- The Dutch data is scraped from nos.nl by html scraping with selenium over chrome browser.
- **The current program needs user to close cookie consent footbar manually when the NOS crawling starts (Chrome browser will start and before the crawling starts, users has 6 seconds to close the cookie consent footbar).**
- Each language's dataset contains 600 articles, randomly (seed=1) splitted into 80% train and 20% test.
- The English data is scraped using search keyword 'abortion' and date range 2022-06-24 to 2022-09-24', three months after the Roe v. Wade overturned.
- The Dutch data is scraped using search keyword 'abortus' and page range 1 to 35 (with invalid links removed) since the data is way less than the English ones.
- The metadata for these two dataset are stored separately in corresponding .csv files

### OUTPUT:
Folders in the directory if they do not exist. Folder Structure:

`├── * input directory`

`│   ├── eng`

`│   │   ├── train`

`│   │   ├── test`

`│   ├── nld`

`│   │   ├── train`

`│   │   ├── test`


### LINKS TO SOURCES AND LICENSES OF THE DOCUMENTS:
- stored in the corresponding .csv files

## Analyzing Data and Creating Plots: run_all_analyses.py
Usage: put under the same folder as the input directory given above, or under the same directory as \eng and \nld.

### OPTIONS
- Using stanza: Use stanza model to process articles. It takes long to load and it may take extremely long to process. Users can select to process articles by themselves or use pre-processed texts.
- Using pre-processed articles: Stanza processed articles are provided at `processed_eng_train.pkl` and `processed_nld_train.pkl`. Users can load these data.
- Using pre-calculated keywords: Keywords takes 5 to 10 minutes to be calculated. Users can load pre calculated keywords from `keys.csv`.

### OUTPUT:
- Statistics are printed on the terminal.
- Plots are stored in \results folder.

