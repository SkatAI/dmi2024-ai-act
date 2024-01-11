'''
- load sample of posts from json file
- auth to openai
    - create openai class
-

'''
import os, re, json, glob, sys
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

# Local
from utils import Clean

# OpenAI
import openai

# weaviate
import weaviate
import weaviate.classes as wvc

# LangChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from langchain.chains import SequentialChain

def postize(d):
    return f'''# {d.title}
by {','.join(d.authors)}

{d.text}'''


def save(df, filename):
    if type(df) == list:
        df = pd.DataFrame(df)
    if 'post' in df.columns:
        df.drop(columns = 'post', inplace = True)
    with open(filename, "w", encoding="utf-8") as f:
        df.to_json(f, force_ascii=False, orient="records", indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_name", help="source_name")
    parser.add_argument("--min_token", help="min_token", default = 500)
    parser.add_argument("--max_token", help="min_token", default = 2000)
    parser.add_argument("--max_posts", help="min_token", default = 100)
    parser.add_argument("--summarize_all", help="all", default = 'no')
    args = parser.parse_args()

    source_name = args.source_name
    min_token = int(args.min_token)
    max_token = int(args.max_token)
    max_posts = int(args.max_posts)
    print(source_name, min_token, max_token)

    input_file = f'./data/proc/json/{source_name}.json'
    tmp_file = f'./data/proc/summary_tmp.json'
    output_file = f'./data/proc/augment/{source_name}_summarized.json'

    data = pd.read_json(input_file)
    # deal with most recent first
    data.sort_values(by = 'date_published', ascending = False, inplace= True)
    data.reset_index(inplace=True, drop = True)
    print(f"Loaded {data.shape} posts")

    # remove already summarized
    if os.path.isfile(output_file):
        summarized = pd.read_json(output_file)
        print(f"checking for already summarized data, found: {summarized.shape[0]} posts")

        if 'pid' in summarized.columns:
            pids = summarized.pid.unique()

        data = data[~data.id.isin(pids)].copy()
        data.reset_index(inplace=True, drop = True)
        print(f"data not summarized: {data.shape}")
    else:
        summarized = None

    if args.summarize_all == 'no':

        cond = (data.token_count >= min_token) & (data.token_count <= max_token)

        print(f"[{min_token},{max_token}] this will summarize {data[cond].shape[0]} posts truncated to {max_posts}")
        is_ok = input('is that ok? (y/n) \n')
        if is_ok != 'y':
            assert 1 == 2,''

        data = data[cond].copy()
    else:
        print(f"[summarize_all] this will summarize all {data.shape[0]} remaining posts truncated to {max_posts}")
        is_ok = input('is that ok? (y/n) \n')
        if is_ok != 'y':
            assert 1 == 2,''


    # print(f"data {author} < 2000 tokens: {data.shape}")

    # set the chain
    prompt_summarize = ChatPromptTemplate.from_template('''
Summarize the text in a concise and parataxis style.
Your output must be short and straight to the point.
Focus on author's intent, logic and arguments.
Instead of saying "the author", explicitly name the author in your summary.

The text:
```
{post}
```
''')

    llm_model = "gpt-4-1106-preview"
    llm_model = "gpt-3.5-turbo-1106"

    llm = ChatOpenAI(temperature=0, model=llm_model)

    chain   = LLMChain(llm=llm, prompt=prompt_summarize,  output_key="response", verbose=False)

    overall_chain = SequentialChain(
        chains=[chain],
        input_variables=["post"],
        output_variables=["response"],
        verbose=True
    )
    data.reset_index(inplace = True, drop = True)

    if data.shape[0] > max_posts:
        data = data[0:max_posts].copy()

    data['post'] = data.apply(lambda d: postize(d), axis = 1)
    data.rename(columns = {'id':'pid'}, inplace = True)

    df = []
    for i, d in data.iterrows():
        print()
        print('--' * 20, i, data.shape[0])
        print(d.post[:200])

        response = overall_chain({ "post": d.post })
        print('--'* 20)
        print(response['response'])
        d = d.to_dict().copy()
        d['summary'] = response['response']
        df.append(d)
        save(df, tmp_file)

    df = pd.DataFrame(df)
    cols = ['pid', 'title', 'url', 'source', 'source_type','initial_source', 'authors', 'tags', 'date_published', 'year_published', 'token_count', 'text', 'summary']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]


    if summarized is not None:
        df = pd.concat([summarized, df])
    df.reset_index(inplace=True, drop = True)
    save(df, output_file)

    # df.loc[df.pid.isnull(),'pid'] = df[df.pid.isnull()].id
    # df.drop(columns= ['id'], inplace= True)
    # cols = list(df.columns)
    # cols.insert(0, cols.pop())
    # df = df[cols]
    # save(df, output_file)
