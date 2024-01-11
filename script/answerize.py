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

# LangChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from langchain.chains import SequentialChain

if __name__ == "__main__":

    client = connect_client()
    vectorizer = which_vectorizer("OpenAI")

    collection_name = "AlignmentKnowledgeBase"
    collection = client.collections.get(collection_name)

    query = "what is The Strangest Thing An AI Could Tell You ?"
    query = "Does Eliezer Yudkowsky believe slower economic growth could benefit FAI development?"

    # generative pipeline
    llm_model = "gpt-4-1106-preview"
    llm = ChatOpenAI(temperature=0.9, model=llm_model)

    prompt_generative_context = ChatPromptTemplate.from_template('''Answer the question based only on the following context:
{context}
Question: {question}''')

    prompt_generative_bare = ChatPromptTemplate.from_template('''Answer the question 
Question: {question}''')


    context_chain   = LLMChain(llm=llm, prompt=prompt_generative_context,  output_key="answer_context", verbose=False)

    overall_context_chain = SequentialChain(
        chains=[context_chain],
        input_variables=["context", "question"],
        output_variables=["answer_context"],
        verbose=True
    )

    bare_chain   = LLMChain(llm=llm, prompt=prompt_generative_bare,  output_key="answer_bare", verbose=False)

    overall_bare_chain = SequentialChain(
        chains=[bare_chain],
        input_variables=["question"],
        output_variables=["answer_bare"],
        verbose=True
    )

    # retrieval
    if True:
        response = collection.query.near_text(
                query = query,
                limit = 1,
                return_metadata = [ 'distance', 'certainty', 'score', 'explain_score', 'is_consistent'],
            )
    else:
        response = collection.query.hybrid(
                query = query,
                query_properties = ['text'],
                limit = 1,
                return_metadata = [ 'distance', 'certainty', 'score', 'explain_score', 'is_consistent'],
            )

    # build the context
    context = []
    for o in response.objects:
        print('--' * 20)
        print(o.properties)
        print(o.metadata)
        context.append(o.properties['text'])

    context = '\n\n'.join(context)

    # generate the answer
    response_context = overall_context_chain({ "context": context, "question": query })
    print("=="* 20, "answer with context")
    print(response_context['answer_context'])
    response_bare = overall_bare_chain({ "question": query })
    print("=="* 20, "answer without context")
    print(response_bare['answer_bare'])

