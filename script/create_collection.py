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


if __name__ == "__main__":

    # connect to weaviate
    cluster_location = "local"
    cluster_location = "cloud-cluster"
    client = connect_client(cluster_location)

    if client.is_live() & client.is_ready():
        print("client is live and ready")
    assert client.is_live() & client.is_ready(), "client is not live or not ready"

    collection_name = "AIActKnowledgeBase"
    create_new_collection = True
    collection_exists = False

    try:
        collection = client.collections.get(collection_name)
    except:
        print("collection already exists", collection_name)
        collection_exists = True
        pass

    if (not create_new_collection) & (collection_exists):
        print('Are you sure you want to re create it ? (type the collection name to continue)')
        x = input()
        if x == collection_name:
            create_new_collection = True
    if create_new_collection:
        print(f"creating new collection {collection_name}")
        # create collection
        vectorizer = which_vectorizer("OpenAI")
        schema = [
            wvc.Property(name='uuid', data_type=wvc.DataType.UUID, skip_vectorization=True),
            wvc.Property(name='author_full', data_type=wvc.DataType.TEXT, skip_vectorization=True),
            wvc.Property(name='token_count', data_type=wvc.DataType.INT, skip_vectorization=True),

            wvc.Property(name='author', data_type=wvc.DataType.TEXT, vectorize_property_name = True),
            wvc.Property(name='title', data_type=wvc.DataType.TEXT, vectorize_property_name = False),
            wvc.Property(name='numbering',data_type=wvc.DataType.TEXT, vectorize_property_name = False),
            wvc.Property(name='text', data_type=wvc.DataType.TEXT, vectorize_property_name = False),
        ]


        collection = create_collection(
                client,
                collection_name,
                vectorizer,
                schema,
            )
    collection = client.collections.get(collection_name)
    print(f"collection {collection_name} has :")
    count_collection(collection)
