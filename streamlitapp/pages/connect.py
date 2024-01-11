import os, re, json, glob, csv
import time, datetime
from datetime import timedelta
from tqdm import tqdm
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
import streamlit_authenticator as stauth

import yaml
from yaml.loader import SafeLoader


if __name__ == "__main__":

    st.title(":red[connect]")

    with open('./credentials_users.yml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

# if st.session_state["authentication_status"] is None:
#      authenticator.login('Login', 'main')

    st.write(f"status: {st.session_state['authentication_status']}")
    # if st.session_state["authentication_status"]:
    #     authenticator.logout('Logout', 'main', key='unique_key')
    #     st.write(f'Welcome *{st.session_state["name"]}*')
    #     st.title('Some content')
    # elif st.session_state["authentication_status"] is False:
    #     st.error('Username/password is incorrect')
    # elif st.session_state["authentication_status"] is None:
    #     st.warning('Please enter your username and password')


    # if authenticator.reset_password('alexis.perrier@gmail.com', 'Reset password'):
    #         st.success('Password modified successfully')
