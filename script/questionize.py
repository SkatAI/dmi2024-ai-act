'''
- load sample of posts from json file
- auth to openai
    - create openai class
-

'''
import os, re, json, glob
import time, datetime
from datetime import timedelta
import pandas as pd
import hashlib
from tqdm import tqdm
pd.options.display.max_columns = 100
pd.options.display.max_rows = 60
pd.options.display.max_colwidth = 100
pd.options.display.precision = 10
pd.options.display.width = 160
pd.set_option("display.float_format", "{:.4f}".format)
import numpy as np
import argparse

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

def postize(title, authors, text):
    return f'''{title}
{','.join(authors)}
{text}'''


def save(df, filename):
    if type(df) == list:
        df = pd.DataFrame(df)
    with open(filename, "w", encoding="utf-8") as f:
        df.to_json(f, force_ascii=False, orient="records", indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_name", help="source_name")
    args = parser.parse_args()

    source_name = args.source_name
    print(source_name)

    input_file = './alignment/data/proc/json/{source_name}_summarized.json'
    tmp_file = './alignment/data/proc/questions_tmp.json'
    output_file = './alignment/data/proc/json/{source_name}_questions.json'

    data = pd.read_json(input_file)
    print(data.shape)

    # remove existing questions
    questionized = pd.read_json(output_file)
    pids = questionized.pid.unique()

    data = data[~data.pid.isin(pids)].copy()
    data.reset_index(inplace=True, drop = True)
    print(f"data not questionized: {data.shape}")



    # set the chain
    # The answers must be concise and short.
    prompt_questionize = ChatPromptTemplate.from_template('''
Your task is to generate a set of 3 question answer pairs about the given text.

The questions must be simple sentences:
- Don't mention "the blog post" or "the text" in the questions.
- The questions must focus on beliefs, values and reasoning of the author

Your output must comply with the following JSON format

```
[{{
    "question": "$FIRST_QUESTION_HERE",
    "answer": "$FIRST_ANSWER_HERE"
}},
{{
    "question": "$SECOND_QUESTION_HERE",
    "answer": "$SECOND_ANSWER_HERE"
}},
{{
    "question": "$THIRD_QUESTION_HERE",
    "answer": "$THIRD_ANSWER_HERE"
}}]
```

Everything contained between ``` must be valid JSON.

Write a series of 3 question/answer pairs, in the specified JSON format, related to the following text:
----------------
```
{post}
```
''')

    llm_model = "gpt-4-1106-preview"
    llm = ChatOpenAI(temperature=0.9, model=llm_model)

    chain   = LLMChain(llm=llm, prompt=prompt_questionize,  output_key="questions", verbose=False)

    overall_chain = SequentialChain(
        chains=[chain],
        input_variables=["post"],
        output_variables=["questions"],
        verbose=True
    )

    data['post'] = data.apply(lambda d: postize(d.title, d.authors, d.summary), axis = 1)

    df = []
    for i, d in data.iterrows():
        print()
        print('--' * 20)
        print(d.post)
        response = overall_chain({ "post": d.post })
        print('--'* 20)
        print(response['questions'])
        d = d.to_dict().copy()
        qas = json.loads(response['questions'].replace("```", "").replace('json', ''))
        for qa in qas:
            qid = hashlib.md5(qa['question'].encode('UTF-8')).hexdigest()
            print('--',qid)
            print("> ",qa['question'])
            print(qa['answer'])
            df.append({
                'pid':d['pid'],
                'qid': qid,
                'title': d['title'],
                'question': qa['question'],
                'answer': qa['answer']
            })
        save(df, tmp_file)

    df = pd.DataFrame(df)

    assert df.shape[0] == 3 * data.shape[0]

    df = pd.concat([questionized, df])

    df.reset_index(inplace=True, drop = True)
    save(df, output_file)
