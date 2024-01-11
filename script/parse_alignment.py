import os, re, json, glob, csv
import time, datetime
from datetime import timedelta
import pandas as pd
import argparse
from tqdm import tqdm
pd.options.display.max_columns = 100
pd.options.display.max_rows = 60
pd.options.display.max_colwidth = 100
pd.options.display.precision = 10
pd.options.display.width = 160
pd.set_option("display.float_format", "{:.4f}".format)
import numpy as np
from collections import Counter
from utils import Clean

def markdownize(d):
    text = f'''
---------
# {d.title}
authors: {', '.join(d.authors)}
'''

    if "date_published" in d.keys():
        text += f'''
date: {d.date_published}'''

    if "tags" in d.keys():
        text += f'''
tags: {', '.join(d.tags)}'''
    text += f'''source: '''
    if "source" in d.keys():
        text += f"{d.source}, "
    if "source_type" in d.keys():
        text += f"{d.source_type}, "
    if "initial_source" in d.keys():
        text += f"{d.initial_source}"
    if "url" in d.keys():
        text += f'''
link: {d.url}'''

    text += f'''
{d.title}
{d.text}
'''
    return text

def save_small(source_name):
    threshold = 1000000
    text = ""
    chars_ = 0
    iter_ = 0
    for i , d in data.iterrows():
        filename = f"./data/proc/medium/{source_name}/{source_name}_{str(iter_).zfill(4)}.txt"
        txt = markdownize(d)
        print(f"[{iter_}] len(txt) {len(txt)} len(text): {len(text)} chars_: {chars_} ")
        if len(txt) + chars_ > threshold:
            # save to file
            print(f"save to: {filename}")
            with open(filename, 'w') as f:
                f.write(text)

            # reset text, chars_
            text = ""
            chars_ = 0
            iter_ += 1
            print(f"increment {iter_}  chars_: {chars_} ")
        text += txt
        chars_ = len(text)






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_name", help="source_name")
    args = parser.parse_args()

    source_name = args.source_name
    print(source_name)

    input_file = f"./data/raw/{source_name}.jsonl"
    output_file_json = f"./data/proc/json/{source_name}.json"
    output_file_csv = f"./data/proc/csv/{source_name}.csv"
    output_file_md = f"./data/proc/txt/{source_name}.txt"

    # Initialize an empty list to store the dictionaries
    data = []

    # Open the JSONL file and read it line by line
    with open(input_file, 'r') as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line))

    data = pd.DataFrame(data)
    print(f"loaded {data.shape} from {input_file}")
    data.dropna(subset = 'text', inplace = True)
    print(f"rm None text -> {data.shape}")

    print("-- clean text")
    data['text'] = data.text.apply(lambda txt : Clean(txt).process())
    data['title'] = data.title.apply(lambda txt : Clean(txt).process())

    # id text with data:image remaining
    data['flag_image'] = data.text.apply(lambda txt : 'data:image' in txt )
    print(f"== {data[data.flag_image].shape[0]} posts with remaining data:image")
    # rm data:image
    data = data[~data.flag_image].copy()
    print(f"remaining {data.shape[0]} posts ")

    # dates
    if 'date_published' in data.columns:
        data['date_published']= pd.to_datetime(data.date_published).dt.date
        data['date_published'] = data['date_published'].apply(str)
    if 'year_published' in data.columns:
        data['year_published']= data.date_published.apply(lambda d : d.year)
    if 'modified_at' in data.columns:
        data['modified_at']= pd.to_datetime(data.modified_at).dt.date

    # count tokens
    print("-- count tokens")
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")

    data['token_count'] = data.text.apply(lambda txt : len(encoding.encode(txt))  )

    print(data.token_count.describe(percentiles = [0.01, 0.05, 0.1, 0.95, 0.99]))

    # filter
    token_limits = {
        'lesswrong': {"min": 40},
        'arbital': {"min": 100},
        'blogs': {"min": 40},
    }
    if source_name not in token_limits.keys():
        token_limits[source_name] = {"min": 40}

    print(f"rm {data[data.token_count <= token_limits[source_name]['min']].shape[0]} rows with less than {token_limits[source_name]['min']} tokens")
    data = data[data.token_count > token_limits[source_name]["min"]].copy()

    data.sort_values(by = 'token_count', ascending = True, inplace = True)
    data.reset_index(inplace = True, drop = True)


    # extract tags
    if 'tags' in data.columns:
        tags = [    tg for ltags in data.tags.values for tg in ltags   ]
        print(f"{len(set(tags))} different tags")
        print(f"Most common tags: \n{Counter(tags).most_common(10)}")

        most_common_tags = [  tag[0] for tag in  Counter(tags).most_common(10) ]

        for tag in most_common_tags:
            print(f"- {tag} {data[data.tags.apply(lambda tgs : tag in tgs   )].shape[0]}")

    # authors
    if 'authors' in data.columns:
        authors = [    auth for lauthors in data.authors.values for auth in lauthors   ]
        print(f"{len(set(authors))} different authors")
        print(f"Most common authors: \n{Counter(authors).most_common(10)}")

        most_common_authors = [  author[0] for author in  Counter(authors).most_common(10) ]

        # for author in most_common_authors:
        #     cond = data.authors.apply(lambda authors : author in authors   )
        #     print(f"- {author} {data[cond].shape[0]}")
        #     # most common tags
        #     if 'tags' in data.columns:
        #         tags = [    tg for ltags in data[cond].tags.values for tg in ltags   ]
        #         print(f"\t{len(set(tags))} different tags")
        #         print(f"\t Most common tags: \n{Counter(tags).most_common(5)}")


    # cols
    cols = ['id', 'title', 'url', 'source', 'source_type','initial_source', 'authors', 'tags', 'date_published', 'year_published', 'token_count', 'text']
    cols = [c for c in cols if c in data.columns]
    data = data[cols]


    print("saving to json")
    with open(output_file_json, "w", encoding="utf-8") as f:
        data.to_json(f, force_ascii=False, orient="records", indent=4)

    assert 1 == 2

    print("saving to csv")
    cols.remove('token_count')
    cols = [c for c in cols if c in data.columns]
    data.to_csv(output_file_csv, index = False, quoting = csv.QUOTE_ALL)




    print("saving to markdown text")
    data['mkdown'] = data.apply(lambda d : markdownize(d), axis = 1 )

    all_texts = " ".join(data.mkdown.values)
    with open(output_file_md, 'w') as f:
        f.write(all_texts)

    # hist
    if False:
        ''' shows that word count is not reliable
            lots of long posts with low word count
        '''

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(1,1, figsize = (9,6))
        plt.hist(data.words, bins = 100)
        plt.title('histograms of words count')
        plt.show()

    if False:
        ''' shows that word count is not reliable
            lots of long posts with low word count
        '''

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(1,1, figsize = (9,6))
        plt.hist(data.year_published, bins = 100)
        plt.title('number of posts per year')
        plt.show()

    if False:
        ''' hist of token_count '''

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(1,1, figsize = (9,6))
        plt.hist(data.token_count, bins = 100)
        plt.title('number of token_count')
        plt.show()
