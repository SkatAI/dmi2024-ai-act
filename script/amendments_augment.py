# usual suspects
import os, re, json, glob
import time, datetime
from datetime import timedelta
import pandas as pd
import argparse
from tqdm import tqdm
pd.options.display.max_columns = 100
pd.options.display.max_rows = 60
pd.options.display.max_colwidth = 160
pd.options.display.precision = 10
pd.options.display.width = 160
pd.set_option("display.float_format", "{:.4f}".format)
import numpy as np

import difflib
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

if __name__ == "__main__":

    sheet_id = "1C06ARBYAyNlOLxiRU2G9uI5BUFOamfo4o5edSmJ9urs"
    data = pd.read_csv(f"https://docs.google.com/spreadsheets/export?id={sheet_id}&format=csv")
    # columns
    data.rename(columns = {'Text proposed by the Commission': 'commission'}, inplace = True)
    data = data[['number', 'group', 'proposers', 'title', 'commission', 'amendment']]
    data['number'] = data.number.apply(lambda d : f"a-{str(d).zfill(4)}")
    data.fillna(
                value = {"commission": '', "amendment": ''}, inplace = True
                )
    # more parsing on title : new
    # diff

    def get_add(text1, text2):
        return ''.join([s.replace('+ ','') for s in  list(difflib.ndiff(text1, text2)) if '+' in s])
    def get_rmd(text1, text2):
        return ''.join([s.replace('- ','') for s in  list(difflib.ndiff(text1, text2)) if '-' in s])

    def find_seq(added, target):
        seq_start =0
        sequences = []
        for i in range(len(added)):
            current_seq = ""
            while (added[seq_start: i] in target) & (i < len(target)):
                current_seq = added[seq_start: i]
                i +=1
            seq_start = i
            if current_seq != '':
                current_seq.replace(';','.')
                sequences.append(current_seq)

        return sequences

    for i, d in tqdm(data.iterrows()):
        added = get_add(d.commission, d.amendment)
        removed = get_rmd(d.commission, d.amendment)
        data.loc[i, 'added'] = '; '.join(find_seq(added, d.amendment))
        data.loc[i, 'removed'] = '; '.join(find_seq(removed, d.commission))

    # count tokens
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    data['commission_token_count'] = data.commission.apply(lambda txt : len(encoding.encode(txt))  )
    data['amendment_token_count'] = data.amendment.apply(lambda txt : len(encoding.encode(txt))  )
    data['added_token_count'] = data.added.apply(lambda txt : len(encoding.encode(txt))  )
    data['removed_token_count'] = data.removed.apply(lambda txt : len(encoding.encode(txt))  )
    import csv
    data.to_csv("./data/commission_amendment_01.csv", quoting = csv.QUOTE_ALL, index = False)
    #