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

    sheet_id = "1Ipu-IFoS8aD386k5inqKli6oSpy7UEUMi9KvZ4q2rV8"
    df = pd.read_csv(f"https://docs.google.com/spreadsheets/export?id={sheet_id}&format=csv")
    sheet_id = "1-F0lOZwGem8rN1cR5-70o_HpjUYijZPwKzhIT4eudzk"
    sheet_id = "1TRlk-g7MyXJcILtP84Bcyj6S2v3XTwzc2WKllzTpWVM"

    # with titles 1214 lines
    sheet_id = "1v0xypQWOa39L574FKQrnNnS5k9C7zfGCUBco00o0eu8"
    sheet_id = "1TRlk-g7MyXJcILtP84Bcyj6S2v3XTwzc2WKllzTpWVM"
    df = pd.read_csv(f"https://docs.google.com/spreadsheets/export?id={sheet_id}&format=csv")



    # df.drop(0, axis = 0, inplace = True)
    df.fillna('', inplace = True)
    cols = ['Commission', 'Council', 'Parliament Adopted','Committee on Culture and Education', 'Committee on Industry, Research and Energy',
       'Committee on Legal Affairs', 'Committee on Transport and Tourism',
       'Committee on the Internal Market and Consumer Protection Committee on Civil Liberties, Justice and Home Affairs',
       'European Conservatives and Reformists Group', 'Group of the European Peoples Party (Christian Democrats)',
       'Group of the Greens/European Free Alliance', 'Group of the Progressive Alliance of Socialists and Democrats in the European Parliament',
       'Identity and Democracy Group', 'Renew Europe Group', 'The Left group in the European Parliament - GUE/NGL']

    acron = {'Commission': 'COMMISSION', 'Council':'COUNCIL', 'Parliament Adopted':'PARLIAMENT',
            'Committee on Culture and Education':'CULT',
            'Committee on Industry, Research and Energy':'ITRE',
            'Committee on Legal Affairs':'JURI',
            'Committee on Transport and Tourism':'TRAN',
            'Committee on the Internal Market and Consumer Protection Committee on Civil Liberties, Justice and Home Affairs':'IMCO-LIBE',
            'European Conservatives and Reformists Group':'ECR',
            'Group of the European Peoples Party (Christian Democrats)':'EPP' ,
            'Group of the Greens/European Free Alliance': 'Greens/EFA',
            'Group of the Progressive Alliance of Socialists and Democrats in the European Parliament':'S&D',
            'Identity and Democracy Group': 'ID',
            'Renew Europe Group': 'Renew',
            'The Left group in the European Parliament - GUE/NGL': 'GUE/NGL'
        }
    data = []
    for i, d in tqdm(df.iterrows()):
        base = {
            'numbering': d.Numbering,
            'title': d.Title,
            'id': d.ID,
        }
        for col in cols:
            if (d[col] != ''):
                base.update({
                    'text': d[col],
                    'author': acron[col],
                    'author_full': col,
                })
                data.append(base.copy())
    data = pd.DataFrame(data)

    assert 1 == 2

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


