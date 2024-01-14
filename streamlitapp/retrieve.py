import os, re, json, glob
import time, datetime
import pandas as pd
pd.options.display.max_columns = 100
pd.options.display.max_rows = 60
pd.options.display.max_colwidth = 100
pd.options.display.precision = 10
pd.options.display.width = 160
pd.set_option("display.float_format", "{:.4f}".format)
import numpy as np

# streamlit
import streamlit as st

# weaviate
from weaviate.classes import Filter

# open AI
from openai import OpenAI

# LangChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from langchain.chains import SequentialChain

# local
from  weaviate_utils_copy import *


class Retrieve(object):

    def __init__(self, query, search_type, gen_model, number_elements, temperature, author, topic):
        cluster_location = "cloud-cluster"
        self.client = connect_client(cluster_location)
        assert self.client is not None
        assert self.client.is_live()

        # retrieval
        self.vectorizer = which_vectorizer("OpenAI")
        self.collection = self.client.collections.get("AIActKnowledgeBase")
        self.search_type = search_type
        self.gen_model = gen_model
        self.response_count_ = number_elements
        self.temperature = temperature
        self.author = author
        self.topic = topic
        if self.topic == '--':
            self.query = query
        else:
            self.query = query.replace("?", ' ') + f", related to {self.topic} ?"


        # generative

        self.prompt_generative_context =ChatPromptTemplate.from_template(
'''You are a political analyst, expert on both AI regulation and European policy making.
# Task
Given a question and a context, your task is to answer the question based on the information contained in the context.

# make sure:
- If the context does not provide an answer to the question, you can then use your global knowledge to answer the question but clearly state that the context does not provide information with regard to the question.
- Focus on the differences in the text between the European union entities: commission, council, parliament as well as the political groups and committees.
- Your answer must be short and concise.

--- Context:
{context}
--- Question:
{question}''')

        self.prompt_generative_bare = ChatPromptTemplate.from_template(
'''You are a political analyst, expert on both AI regulation and European policy making.
# Task
Your task is to answer the question below.

# make sure:
- clearly state if you can't find the answer to the question. Do not try to invent an answer.
- Focus on the differences in the text between the European union entities: commission, council, parliament as well as the political groups and committees.
- Your answer must be short and concise. One line if possible
--- Question:
{question}''')



        self.llm = ChatOpenAI(temperature=self.temperature, model=self.gen_model)
        self.context_chain   = LLMChain(llm=self.llm, prompt=self.prompt_generative_context,  output_key="answer_context", verbose=False)

        self.overall_context_chain = SequentialChain(
            chains=[self.context_chain],
            input_variables=["context", "question"],
            output_variables=["answer_context"],
            verbose=True
        )

        self.bare_chain   = LLMChain(llm=self.llm, prompt=self.prompt_generative_bare,  output_key="answer_bare", verbose=False)

        self.overall_bare_chain = SequentialChain(
            chains=[self.bare_chain],
            input_variables=["question"],
            output_variables=["answer_bare"],
            verbose=True
        )

    def search(self):
        if self.author!= "--":
            filters = Filter("author").equal(self.author)
        else:
            filters = None
        if self.search_type == "hybrid":
            self.response = self.collection.query.hybrid(
                    query=self.query,
                    query_properties = ['text'],
                    filters= filters,
                    limit=self.response_count_,
                    return_metadata = ['score', 'explain_score', 'is_consistent'],
                )
        if self.search_type == "near_text":
            self.response = self.collection.query.near_text(
                    query= self.query,
                    filters= filters,
                    limit=self.response_count_,
                    return_metadata = ['distance','certainty'],
                )
        self.get_context()

    def generate_answer_with_context(self):
        self.response_context = self.overall_context_chain({ "context": self.context, "question": self.query })
        self.answer_with_context = self.response_context['answer_context']

    def generate_answer_bare(self):
        self.response_bare = self.overall_bare_chain({ "question": self.query })
        self.answer_bare = self.response_bare['answer_bare']

    def get_context(self):
        texts = []
        for i in range(self.response_count_):
            prop = self.response.objects[i].properties
            text = "---"
            text += ' - '.join([prop['title'], prop['numbering'], prop['author'] ])
            text += "\n"
            text += prop['text']

            texts.append(text)
        self.context = '\n'.join(texts)

    # format
    def format_metadata(self, i):
        metadata_str =[]
        if self.search_type == "hybrid":
            metadata_str =f"**score**: {np.round(self.response.objects[i].metadata.score, 4)} "
        elif self.search_type == "near_text":
            metadata_str = f"distance: {np.round(self.response.objects[i].metadata.distance, 4)} certainty: {np.round(self.response.objects[i].metadata.certainty, 4)} "
        st.write(metadata_str)

    def retrieved_title(self, i):
        prop = self.response.objects[i].properties
        # headlinize
        # title = prop['text'].split('\n')[0].strip()
        title = ' - '.join([prop['title'], prop['numbering'], prop['author'] ])
        return  f"**{title}**"

    def format_properties(self, i):
        prop = self.response.objects[i].properties
        # remaining_words_count = len(prop['text'].split(' ')) - 100
        # text = ' '.join(prop['text'].split(' ')[:100])
        # headlinize
        # text = prop['text'].split("\n")
        # text[0] = f"**{text[0].strip()}**"
        # text = '\n'.join(text).replace('\n', '  \n')
        st.write(prop['text'].strip())

        # url = f"  [{prop['pid']}]({prop['url']}) "
        # authors = ', '.join(prop['authors'])
        # if 'tags' in prop.keys():
        #     tags = ', '.join(prop['tags'])
        # else:
        #     tags = ''
        # st.caption(' - '.join([prop['source'], prop['source_type'], prop['text_type'], authors, url, tags]))

    def save(self):
        # print(self.__dict__.keys())
        os.write(1,bytes("--"*20 + "\n", 'utf-8'))
        os.write(1,bytes(f"query: {self.query}\n" , 'utf-8'))
        os.write(1,bytes(f"search_type: {self.search_type}\n" , 'utf-8'))
        os.write(1,bytes(f"gen_model: {self.gen_model}\n" , 'utf-8'))
        os.write(1,bytes(f"response_count_: {self.response_count_}\n" , 'utf-8'))
        os.write(1,bytes(f"temperature: {self.temperature}\n" , 'utf-8'))
        os.write(1,bytes(f"author: {self.author}\n" , 'utf-8'))
        os.write(1,bytes(f"answer with context:\n {self.answer_with_context}\n" , 'utf-8'))
        os.write(1,bytes("--"*20 + "\n\n", 'utf-8'))
        pass


def perform_search(query, search_type, gen_model):
    '''
    - vectorize query
        - client, vectorizer etc
        - store query
    - near text search
    - prompt
    - answer generation
    '''
    retr = Retrieve(query, search_type, gen_model)
    retr.search()

    return retr


