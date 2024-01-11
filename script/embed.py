'''
TODO:
- add initial_source
- when getting pids, filter by source
- dependency on limit for fetch_objects
'''
import os, re, json, glob
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
from weaviate_utils import *


# OpenAI
import openai
import tiktoken

# weaviate
import weaviate
import weaviate.classes as wvc
from weaviate.classes import Filter

# LangChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from langchain.chains import SequentialChain

def postize(title, authors, text):
    return f'''{title}
{','.join(authors)}
{text}'''
import uuid

create_new_collection = False

if __name__ == "__main__":
    input_file = "./data/full_amendments_jan10-1637.csv"

    data = pd.read_csv(input_file)
    print("-- loaded ",data.shape[0], "items")

    # clean up data
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    data['token_count'] = data.text.apply(lambda txt : len(encoding.encode(txt))  )
    data.fillna('', inplace = True)
    data['numbering'] = data['numbering'].apply(lambda d : d.strip())
    data['uuid'] = [ str(uuid.uuid4()) for i in range(len(data))  ]
    data = data[['uuid','author_full', 'token_count', 'title', 'numbering', 'author', 'text']].copy()

    assert 1 == 2


    # wvc.Property(name='author_full', data_type=wvc.DataType.TEXT_ARRAY, skip_vectorization=True),
    # wvc.Property(name='token_count', data_type=wvc.DataType.INT, skip_vectorization=True),
    # wvc.Property(name='title', data_type=wvc.DataType.TEXT, vectorize_property_name = False),
    # wvc.Property(name='numbering',data_type=wvc.DataType.TEXT, vectorize_property_name = False),
    # wvc.Property(name='author', data_type=wvc.DataType.TEXT_ARRAY, vectorize_property_name = True),
    # wvc.Property(name='text', data_type=wvc.DataType.TEXT, vectorize_property_name = False),


    # connect to weaviate
    client = connect_client(location="cloud-cluster")
    collection_name = "AIActKnowledgeBase"

    collection = client.collections.get(collection_name)

    batch_result = collection.data.insert_many(data.to_dict(orient = 'records'))
    if batch_result.has_errors:
        print(batch_result.errors)
        raise "stopping"

    count_collection(collection)
