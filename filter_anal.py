import sys
import os
import csv
import json
import spacy
from spacy.lang.en import English
from typing import Dict, Tuple, Sequence, List, Callable
import argparse

nlp = spacy.load("en_core_web_sm")
spacy_tokenizer = English().Defaults.create_tokenizer(nlp)

import numpy
import torch
import h5py
from pytorch_pretrained import BertTokenizer, BertModel, BertForMaskedLM
from pytorch_pretrained import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
# from allennlp.commands.elmo import ElmoEmbedder
# from allennlp.data.tokenizers.token import Token
# from allennlp.common.tqdm import Tqdm
import tqdm

def main():
    parser = argparse.ArgumentParser(description="filter analogy file based on word filter")
    parser.add_argument('--input', default="/home/tom/web_data/analogy/EN-GOOGLE/EN-GOOGLE.txt",
                         help='input file')
    parser.add_argument('--output', default="anal/filtered.txt",
                         help='output file')
    parser.add_argument('--word-filter', default=False,
                         help='keep only sentences that match set of words in filter')
    args = parser.parse_args()

    input_file = args.input
    output_file_parts = args.output.split(".")
    words_used_output = f"{output_file_parts[0]}_words.{output_file_parts[1]}"

    word_filter_set = set()
    words_used = set()

    with open(args.word_filter) as file_in:
      for line in file_in:
        word = line.strip()
        word_filter_set.add(word)

    with open(args.input, "r") as f:
      L = f.read().splitlines()

    num_lines = 0
    num_passed = 0
    num_filtered = 0
    with open(args.output, 'w') as f:
      for l in L:
        if l.startswith(":"):
          f.write(f"{l}\n")
        else:
          # check for set inclusion
          words_on_line = set(l.lower().split())
          if words_on_line.issubset(word_filter_set):
            # print("words {} found {}".format(words_on_line, words_on_line.intersection(word_filter_set)))
            f.write(f"{l}\n")
            words_used.update(words_on_line)
            num_passed += 1
            # sys.exit(0)
          else:
            num_filtered += 1
        num_lines += 1

    with open(words_used_output, 'w') as f:
        f.write("\n".join(map(str, sorted(words_used))))

    print("Summary: saved {} analogies out of {} ({} filtered, {} words)".format(num_passed, num_lines, num_filtered, len(words_used)))

if __name__ == '__main__':
    main()
