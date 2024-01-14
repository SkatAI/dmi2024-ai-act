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

import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

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

    # st.subheader(":blue[EIFFEL:] :orange[European Intelligence Framework For Evaluating Legislation]")
    st.subheader(":blue[EIFFEL:] :orange[AI-Act process and content analysis]")
    st.caption("by [Université Gustave Eiffel](https://www.univ-gustave-eiffel.fr/en/)")

    # st.write("best suited for questions such as : on topic <topic>, what differences do you see between the groups")
    # status = st.session_state['authentication_status']
    # st.write(f"status: {status}")
    # st.session_state["authentication_status"] = True
    # ----------------------------------------------------------------------------
    # Sidebar
    # ----------------------------------------------------------------------------
    with st.sidebar:
        # st.image("logo-gustave-eiffel.png")
        # if st.session_state["authentication_status"]:

        gen_model_options = ["gpt-3.5-turbo-1106","gpt-4-1106-preview"]
        gen_model = st.selectbox(
                "Generative model",
                gen_model_options,
                index = 0 if st.session_state.get("gen_model_key") is None else gen_model_options.index(st.session_state.get("gen_model_key")),
                key = "gen_model_key"
            )

        # add author + title + numbering
        # author
        author_options = ['--','COMMISSION', 'COUNCIL',  'PARLIAMENT', 'CULT', 'ECR', 'EPP', 'IMCO-LIBE', 'ITRE', 'JURI', 'TRAN', 'GUE/NGL','Greens/EFA', 'ID', 'Renew', 'S&D']
        author = st.selectbox("Authored by", author_options,
                index = 0 if st.session_state.get("author_key") is None else author_options.index(st.session_state.get("author_key")),
                key = "author_key"
            )

        # topics
        topics_options = topics = ['--','AI Conformity Assessment and Certification', 'AI Impact on Fundamental Rights and Democracy', 'AI Innovation and Support for SMEs and Start-ups', 'AI Regulatory Framework Development', 'AI Standardization and Harmonization', 'AI System Transparency and Accountability', 'AI Systems in Financial and Medical Sectors', 'AI Systems in Public Sector and Law Enforcement', 'Data Protection and Privacy in AI Systems', 'Ethical and Trustworthy AI Development', 'International and Cross-border AI Regulation', 'Market Surveillance and Enforcement of AI Regulation', 'Stakeholder Engagement and Public Consultation in AI Regulation']
        topic = st.selectbox("Topic", topics_options,
                index = 0 if st.session_state.get("topic_key") is None else topics_options.index(st.session_state.get("topic_key")),
                key = "topic_key"
            )

        st.divider()
        search_type_options = ["hybrid", "near_text"]
        search_type = st.selectbox("Search type", search_type_options,
                index = 0 if st.session_state.get("search_type_key") is None else search_type_options.index(st.session_state.get("search_type_key")),
                key = "search_type_key"
            )

        number_elements_options = [1,2,3,4,5,6,7,8,9,10]
        number_elements = st.selectbox("Number of retrieved elements", number_elements_options,
                index = 4 if st.session_state.get("number_elements_key") is None else number_elements_options.index(st.session_state.get("number_elements_key")),
                key = "number_elements_key"
            )

        # temperature
        temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, step=0.1,
                            value= 0.5 if st.session_state.get("search_temperature_key") is None else st.session_state.get("search_temperature_key"),
                            key = "search_temperature_key")
        # authenticator.logout('Logout', 'main', key='unique_key')

        st.divider()
        st.caption("[github: SkatAI/dmi2024-ai-act](https://github.com/SkatAI/dmi2024-ai-act)")


    # ----------------------------------------------------------------------------
    # Main query input
    # ----------------------------------------------------------------------------

    # if st.session_state["authentication_status"]:
    sc1, sc2 = st.columns([3,1])
    with sc1:
        with st.form('search_form', clear_on_submit = False):
            sc3, sc4 = st.columns([8,1])
            with sc3:
                search_query = st.text_input("Your question:", key="query_input" )
                search_scope = st.checkbox("Show the answer generated without context")

            with sc4:
                st.write(' ')
                search_button = st.form_submit_button(label="Ask")

    if not search_button:
        st.write("This RAG/GPT AI allows you to explore the EU AI-act in its different versions (Commission, Council and Parliament) as well as related amendments from political groups (Renew, Greens/EFA, EPP, ID, ...) and committees (CULT, TRAN, IMCO, ..)")
        st.write("Some questions you can ask the EIFFEL GPT:")
        st.write("- What are the main topics addressed by the Greens/EFA group ?")
        st.write("- How does the JURI committee and the Council differ on biometric systems ?")
        st.write("- How do the JURI and ITRE committees diverge regarding discriminatory effects of AI systems ?")
        st.write("- What are the main contributions of the Renew political group ? (GPT4, 10 docs)")
        st.write("- How does the commission enforce  mitigation of high risks AI systems")
        st.write("- Describe the risk management system for providers")
        st.write("- How does the CULT committee and the Council differ on biometric systems ?")

    # ----------------------------------------------------------------------------
    # Search results
    # ----------------------------------------------------------------------------
    if search_button:
        retr = Retrieve(search_query, search_type, gen_model, number_elements, temperature, author, topic)
        retr.search()

        # with sc2:

            # st.caption(", ".join([search_type, gen_model, str(number_elements), str(temperature), author, topic]))
            # st.caption(f"Your question was:   \n**{search_query}**  \nwith search type: **{search_type}**, ** content. {gen_model}, number_elements {number_elements}, temperature: {temperature}, author: {author}, topic {topic}")
        if search_scope:
            st.subheader("Answer without context:")
            retr.generate_answer_bare()
            _, col2 = st.columns([1, 15])
            with col2:
                st.markdown(f"<em>{retr.answer_bare}</em>", unsafe_allow_html=True)

        st.subheader("Answer with retrieved documents")
        retr.generate_answer_with_context()
        _, col2 = st.columns([1, 15])
        with col2:
            st.markdown(f"<em>{retr.answer_with_context}</em>", unsafe_allow_html=True)

        retr.save()
        st.header("Retrieved documents")

        for i in range(len(retr.response.objects)):
            with st.expander(retr.retrieved_title(i)):
                col1, col2 = st.columns([1, 15])
                with col1:
                    st.subheader(f"{i+1})")
                with col2:
                    retr.format_properties(i)
                    retr.format_metadata(i)
                    st.divider()


    # st.divider()
    # with st.expander("Context given wo the prompt: "):
    #     st.caption(retr.context)

    # ----------------------------------------------------------------------------
    # Connection form
    # ----------------------------------------------------------------------------
    # if (st.session_state["authentication_status"] is None) | (not st.session_state["authentication_status"] ) :
    #     authenticator.login('Login', 'main')
    #     st.warning("Please enter your username and password")
