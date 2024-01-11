import os, re, json
import time, datetime
import pandas as pd
import numpy as np

import weaviate
import weaviate.classes as wvc

lclient = weaviate.connect_to_local(
            port=8080,
            grpc_port=50051,
            headers={
                "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"],
            }
        )



client = weaviate.connect_to_wcs(
            cluster_url="https://6q45jbvsfo4cmhzrpjwoq.c1.europe-west3.gcp.weaviate.cloud",
            auth_credentials=weaviate.AuthApiKey(os.environ["WEAVIATE_KEY"]),
            headers={
                "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"],
            }
    )

client = weaviate.connect_to_wcs(
        cluster_id="6q45jbvsfo4cmhzrpjwoq",
        auth_credentials=weaviate.AuthApiKey(os.getenv("WEAVIATE_KEY")),
        headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
    )

auth_config = weaviate.AuthApiKey(api_key="pubTduiMmWZG9KY4lyBOysnRvvc5QSIFyrVH")
client = weaviate.Client(
    url="https://6q45jbvsfo4cmhzrpjwoq.c1.europe-west3.gcp.weaviate.cloud",
    auth_client_secret=auth_config,
    additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)
