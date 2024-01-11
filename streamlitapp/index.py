'''
Next
- add list of questions
    - for known questions add expected answer
    - compare with
- filter by summary vs full_post
- question in search bar
- generative answer
- add model to search
'''
# usual suspects
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
# import streamlit_authenticator as stauth

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
from retrieve import Retrieve


import yaml
from yaml.loader import SafeLoader


if __name__ == "__main__":

    st.set_page_config(
        page_title="AI-act Knowledge Base",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={"About": "Knowledge Base on AI-act dataset - RAG"}
    )

    # with open('./credentials_users.yml') as file:
    #     config = yaml.load(file, Loader=SafeLoader)

    # authenticator = stauth.Authenticate(
    #     config['credentials'],
    #     config['cookie']['name'],
    #     config['cookie']['key'],
    #     config['cookie']['expiry_days'],
    #     config['preauthorized']
    # )

    st.title(":blue[EU - AI-act Enquirer]")

    # st.write("best suited for questions such as : on topic <topic>, what differences do you see between the groups")
    # status = st.session_state['authentication_status']
    # st.write(f"status: {status}")
    # st.session_state["authentication_status"] = True
    # ----------------------------------------------------------------------------
    # Sidebar
    # ----------------------------------------------------------------------------
    with st.sidebar:

        # if st.session_state["authentication_status"]:

        gen_model_options = ["gpt-3.5-turbo-1106","gpt-4-1106-preview"]
        gen_model = st.selectbox(
                "Generative model",
                gen_model_options,
                index = 0 if st.session_state.get("gen_model_key") is None else gen_model_options.index(st.session_state.get("gen_model_key")),
                key = "gen_model_key"
            )

        search_type_options = ["hybrid", "near_text"]
        search_type = st.selectbox("Search type", search_type_options,
                index = 0 if st.session_state.get("search_type_key") is None else search_type_options.index(st.session_state.get("search_type_key")),
                key = "search_type_key"
            )
        # add author + title + numbering
        # author
        author_options = ['--','COMMISSION', 'COUNCIL',  'PARLIAMENT', 'CULT', 'ECR', 'EPP', 'IMCO-LIBE', 'ITRE', 'JURI', 'TRAN', 'GUE/NGL','Greens/EFA', 'ID', 'Renew', 'S&D']
        author = st.selectbox("Authored by", author_options,
                index = 0 if st.session_state.get("author_key") is None else author_options.index(st.session_state.get("author_key")),
                key = "author_key"
            )

        number_elements_options = [1,2,3,4,5,6,7,8,9,10]
        number_elements = st.selectbox("Number of retrived elements", number_elements_options,
                index = 4 if st.session_state.get("number_elements_key") is None else number_elements_options.index(st.session_state.get("number_elements_key")),
                key = "number_elements_key"
            )

        # temperature
        temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, step=0.1,
                            value= 0.5 if st.session_state.get("search_temperature_key") is None else st.session_state.get("search_temperature_key"),
                            key = "search_temperature_key")
        # authenticator.logout('Logout', 'main', key='unique_key')


    # ----------------------------------------------------------------------------
    # Main query input
    # ----------------------------------------------------------------------------

    # if st.session_state["authentication_status"]:
    sc1, sc2 = st.columns([3,1])
    with sc1:
        with st.form('search_form', clear_on_submit = False):
            search_query = st.text_input("Your query:", key="query_input" )
            search_button = st.form_submit_button(label="OK")

    if not search_button:
        st.write("Explore the EU AI-act (Commission, Council and Parliament), related amendments from political groups (Renew, Greens/EFA, EPP, ID, ...) and committees (CULT, TRAN, IMCO, ..)")
        st.write("For instance:")
        st.write("- What are the main topics addressed by the Greens/EFA group ?")
        st.write("- How does the JURI committee and the Council differ on biometric systems ?")
        st.write("- How do the JURI and ITRE committees diverge regarding discriminatory effects of AI systems ?")
        st.write("- What are the main contributions of the Renew political group ? (GPT4, 10 docs)")



    # ----------------------------------------------------------------------------
    # Search results
    # ----------------------------------------------------------------------------
    if search_button:
        retr = Retrieve(search_query, search_type, gen_model, number_elements, temperature, author)
        retr.search()

        with sc2:
            st.caption(f"Your question was:   \n**{search_query}**  \nwith search type: **{search_type}**, ** content. {gen_model}")
            # st.caption(f"Your question was:   \n**{search_query}**  \nwith search type: **{search_type}**, ** content. {gen_model}, number_elements {number_elements}, temperature: {temperature}, author: {author}")

        st.subheader("Answer without context:")
        retr.generate_answer_bare()
        _, col2 = st.columns([1, 11])
        with col2:
            st.markdown(f"<em>{retr.answer_bare}</em>", unsafe_allow_html=True)

        st.subheader("Answer with retrieved elements")
        retr.generate_answer_with_context()
        _, col2 = st.columns([1, 11])
        with col2:
            st.markdown(f"<em>{retr.answer_with_context}</em>", unsafe_allow_html=True)

        retr.save()
        st.header("Retrieved elements")

        for i in range(len(retr.response.objects)):
            with st.expander(retr.retrieved_title(i)):
                col1, col2 = st.columns([1, 11])
                with col1:
                    st.subheader(f"{i+1})")
                with col2:
                    retr.format_properties(i)
                    retr.format_metadata(i)
                    st.divider()
        st.divider()
        with st.expander("Context given wo the prompt: "):
            st.caption(retr.context)

    # ----------------------------------------------------------------------------
    # Connection form
    # ----------------------------------------------------------------------------
    # if (st.session_state["authentication_status"] is None) | (not st.session_state["authentication_status"] ) :
    #     authenticator.login('Login', 'main')
    #     st.warning("Please enter your username and password")
