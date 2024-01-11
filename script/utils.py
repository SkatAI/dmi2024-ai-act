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


class Clean(object):

    def __init__(self, original_text):
        self.original_text = original_text
        self.text = self.original_text


    def add_line_return_between_paragraphs(self):
        self.text = re.sub(r'\.\n([A-Z])',r'.\n\n\1', self.text)

    def remove_numbers_end_of_words(self):
        self.text = re.sub(r'\b([a-z]+)\d{1,2}\b', r'\1', self.text)

    def rm_png(self):
        self.text = re.sub(r'\(data:image[\s\S]*?\n\)', '', self.text)



    def rm_noise(self):
        self.text = self.text.replace("EN EN", '')
        self.text = self.text.replace("", ' ')
        self.text = self.text.replace("\xa0", ' ')
        self.text = self.text.replace("\n>", ' ')
        self.text = self.text.replace("*", ' ')
        self.text = self.text.replace("#", ' ')
        self.text = self.text.replace("^", ' ')
        self.text = self.text.replace("\'", "'")
        self.text = self.text.replace("“", ' ')
        self.text = self.text.replace("”", ' ')
        self.text = self.text.replace("[]", ' ')
        self.text = self.text.replace("[Image]", ' ')
        self.text = self.text.replace("<|endoftext|>", 'endoftext')
        self.text = self.text.replace("<|endofprompt|>", 'endofprompt')
        self.text = self.text.replace("<|endofreply|>", 'endofreply')

    def rm_urls(self):
        self.text = re.sub(
            r'\(http[^)]*\)',
            '',
            self.text
        )

    def rm_numbers_in_square_brackets(self):
        self.text = re.sub(
            r'\[\d]*\]',
            '',
            self.text
        )

    def rm_hashtag_in_parenthesis(self):
        self.text = re.sub(
            r'\(#[^)]*\)',
            '',
            self.text
        )

    def trim_whitespaces(self):
        self.text = re.sub(
            r' +',
            ' ',
            self.text
        )

    def trim_too_many_line_returns(self):
        self.text = re.sub(
            r'\n{2,}',
            '\n\n',
            self.text
        )

    def rm_too_many_consecutives(self):
        self.text = re.sub(
            r'(=|-){2,}',
            '',
            self.text
        )



    def rm_square_brackets(self):
        self.text = re.sub(
            r'\[([^]]*)\]' ,
            r'\1',
            self.text
        )

    def rm_all_within_curly_braces(self):
        self.text = re.sub(
            r'\{([^]]*)\}' ,
            '\n',
            self.text
        )

    def rm_parenthesis_around_text(self):
        self.text = re.sub(
            r'\((.*?)\)' ,
            r' \1 ',
            self.text
        )

    def rm_line_starting_with(self):
        self.text = re.sub(r'^(?:@|<|\.M|\.m).*?$', '', self.text, flags=re.MULTILINE)

    def rm_line_starting_with_data(self):
        self.text = re.sub(r'^! data:<;.*?$', '', self.text, flags=re.MULTILINE)


    def process(self):
        self.add_line_return_between_paragraphs()
        self.rm_urls()
        self.rm_square_brackets()
        self.remove_numbers_end_of_words()
        self.rm_noise()
        self.rm_line_starting_with()
        self.rm_too_many_consecutives()
        self.rm_hashtag_in_parenthesis()
        self.rm_numbers_in_square_brackets()
        self.rm_parenthesis_around_text()
        self.rm_all_within_curly_braces()
        self.rm_parenthesis_around_text()
        self.rm_noise()
        self.rm_png()
        self.trim_whitespaces()
        self.trim_too_many_line_returns()
        self.text = self.text.strip()
        return self.text


if __name__ == "__main__":



    # add_line_return_between_paragraphs
    text = '''
Hello world.
This is a new line.
1. and some footnote
    '''.strip()

    expected = '''
Hello world.

This is a new line.
1. and some footnote
    '''.strip()

    cln = Clean(text)
    cln.process()
    assert expected == cln.text

    # rm_urls
    text = '''
Hello world.
link: [hello](https://alexis.com?url=1&e=wer)
1. and some footnote
    '''.strip()

    expected = '''
Hello world.
link: hello
1. and some footnote
    '''.strip()

    cln = Clean(text)
    cln.process()
    assert expected == cln.text



    text = '''
<http://openai.com>

> We’ll need to invest billions of dollars in upcoming years into large-scale cloud compute, attracting and retaining talented people, and building AI supercomputers.
'''.strip()

    expected = '''
 We’ll need to invest billions of dollars in upcoming years into large-scale cloud compute, attracting and retaining talented people, and building AI supercomputers.
 '''.strip()
    cln = Clean(text)
    cln.process()
    assert expected == cln.text

    if False:
        print(os.getcwd())
        jsonl_file_path = "./alignment/data/raw/lesswrong.jsonl"

        # Initialize an empty list to store the dictionaries
        data = []

        # Open the JSONL file and read it line by line
        with open(jsonl_file_path, 'r') as jsonl_file:
            for line in jsonl_file:
                data.append(json.loads(line))

        data = pd.DataFrame(data)

        text = data.loc[440].text
        print(text[:1000])
        cln = Clean(text)
        cln.process()
        print("== -- _- "* 20)
        print(cln.text[:2000])
