'''

extract text from PDf using pdfplumber
'''


# usual suspects
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

# pdf
import pdfplumber

def add_line_return_between_paragraphs(text):
    pattern = re.compile(r'\.\n([A-Z])')
    text = pattern.sub(r'.\n\n\1', text)
    return text

def remove_numbers_end_of_words(text):
    # TODO link the number to the corresponding footnote to tag the footnote
    return re.sub(r'\b([a-z]+)\d{1,2}\b', r'\1', text)

def remove_lines_with_only_numbers(text):
    # TODO: tag number as page number
    pattern = re.compile(r'^\d+$', re.MULTILINE)
    cleaned_text = pattern.sub('', text)
    return cleaned_text

def add_line_return_before_lines_start_with_number(text):
    # to single out header numbering
    # TODO check relevant for all sections
    pattern = re.compile(r'\n(\d)')
    text = pattern.sub(r'\n\n\1', text)
    return text

def rm_noise(text):
    text = text.replace("EN EN", '')
    text = text.replace("", '* ')
    return text

def tag_footnote(text):
    # lines that start with numbers followed by space : add marker
    # '1 ' not '1.'
    return re.sub(r'\n(\d+) ', r'XXXXXX \1 ', text)

def add_lr_around_article(text):
    return re.sub(r'\n\((\d+)\) ', r'\n\n(\1) ', text)


if __name__ == "__main__":

    file = "JURI-AD-719827_EN.pdf"
    outfile = f"{os.path.splitext(file)[0]}.txt"
    filepath = "./data/pdf"
    outpath = "./data/txt"

    pdf = pdfplumber.open(os.path.join(filepath,file))

    print(f"Document {file} has {len(pdf.pages)} pages")

    text = []
    for i in tqdm(range(len(pdf.pages))):
    # for i in tqdm(range(40)):
        txt = pdf.pages[i].extract_text()
        txt = add_line_return_between_paragraphs(txt)
        txt = remove_numbers_end_of_words(txt)
        txt = remove_lines_with_only_numbers(txt)
        txt = add_line_return_before_lines_start_with_number(txt)
        txt = rm_noise(txt)
        txt = tag_footnote(txt)
        txt = add_lr_around_article(txt)

        text.append(txt)

    text = '\n-----------\n'.join(text)

    with open(  os.path.join(outpath, outfile) , 'w') as f:
        f.write(text)

    print(f"text written to {os.path.join(outpath, outfile)} ")

