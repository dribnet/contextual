import os
import csv
import json
import spacy
import sys
from spacy.lang.en import English
from typing import Dict, Tuple, Sequence, List, Callable
import argparse

nlp = spacy.load("en_core_web_sm")
spacy_tokenizer = English().Defaults.create_tokenizer(nlp)

import numpy
import torch
import h5py
import pytorch_pretrained
import transformers
import tqdm
import re

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from transformers import AutoTokenizer

class Vectorizer:
  """
  Abstract class for creating a tensor representation of size (#layers, #tokens, dimensionality)
  for a given sentence.
  """
  def vectorize(self, sentence: str) -> numpy.ndarray:
    """
    Abstract method for tokenizing a given sentence and return embeddings of those tokens.
    """
    raise NotImplemented

  def make_hdf5_file(self, sentences: List[str], out_fn: str) -> None:
    """
    Given a list of sentences, tokenize each one and vectorize the tokens. Write the embeddings
    to out_fn in the HDF5 file format. The index in the data corresponds to the sentence index.
    """
    sentence_index = 0

    with h5py.File(out_fn, 'w') as fout:
      for sentence in tqdm.tqdm(sentences):
        try:
          embeddings = self.vectorize(sentence)
          fout.create_dataset(str(sentence_index), embeddings.shape, dtype='float32', data=embeddings)
          sentence_index += 1
        except IndexError:
          print("Strange IndexError - skipping {}".format(sentence))

class Bert(Vectorizer):
  def __init__(self, MODEL_NAME=None):
    if MODEL_NAME is None:
      MODEL_NAME = 'bert-base-cased'
    print("Setting up bert model {}".format(MODEL_NAME))
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    config = transformers.BertConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
    self.model = transformers.BertModel.from_pretrained(MODEL_NAME, config=config)

  def vectorize(self, sentence: str) -> numpy.ndarray:
    """
    Return a tensor representation of the sentence of size (13 layers, num tokens, 768 dim).
    Even though there are only 12 layers in GPT2, we include the input embeddings as the first
    layer (for a fairer comparison to ELMo).
    """
    # tokens = self.tokenizer.encode(sentence, add_special_tokens=True)
    result = self.tokenizer.encode_plus(sentence, return_token_type_ids=True, return_tensors="pt")
    # input_ids = torch.tensor(tokens).unsqueeze(0)  # Batch size 1
    outputs = self.model(result.input_ids)
    output_embeddings = outputs[2]
    # embeddings = torch.stack(output_embeddings, dim=0).squeeze()[:,1:-1,:]
    embeddings = torch.stack(output_embeddings, dim=0).squeeze()[:,:,:]
    embeddings = embeddings.detach().numpy()
    # print("NEW", embeddings.shape)            
    return embeddings

class GPT2(Vectorizer):
  def __init__(self):
    MODEL_NAME = "gpt2"
    print("Setting up GPT model {}".format(MODEL_NAME))
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    self.tokenizer.pad_token = self.tokenizer.eos_token
    config = GPT2Config.from_pretrained(MODEL_NAME, output_hidden_states=True)
    self.model = GPT2Model.from_pretrained(MODEL_NAME, config=config)

  def vectorize(self, sentence: str) -> numpy.ndarray:
    result = self.tokenizer.encode_plus(sentence, return_token_type_ids=True, return_tensors="pt")
    outputs = self.model(result.input_ids)
    output_embeddings = outputs[2]
    embeddings = torch.stack(output_embeddings, dim=0).squeeze()[:,:,:]
    embeddings = embeddings.detach().numpy()
    return embeddings

class Roberta(Vectorizer):
  def __init__(self):
    MODEL_NAME = "roberta-base"
    print("Setting up Roberta model {}".format(MODEL_NAME))
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    config = transformers.RobertaConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
    self.model = transformers.RobertaModel.from_pretrained(MODEL_NAME, config=config)

  def vectorize(self, sentence: str) -> numpy.ndarray:
    result = self.tokenizer.encode_plus(sentence, return_token_type_ids=True, return_tensors="pt")
    outputs = self.model(result.input_ids)
    output_embeddings = outputs[2]
    embeddings = torch.stack(output_embeddings, dim=0).squeeze()[:,:,:]
    embeddings = embeddings.detach().numpy()
    return embeddings

class T5Base(Vectorizer):
  def __init__(self, MODEL_NAME=None, sentence_prefix=None):
    if MODEL_NAME is None:
      MODEL_NAME = 't5-base'
    if sentence_prefix is None:
      sentence_prefix = 'sst2 sentence: '
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    config = T5Config.from_pretrained(MODEL_NAME, output_hidden_states=True)
    self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)
    self.MODEL_NAME = MODEL_NAME
    self.sentence_prefix = sentence_prefix
    print("Setup T5 model {} with prefix [{}]".format(self.MODEL_NAME, self.sentence_prefix))
    # self.sentence_prefix = "sst2 sentence: "

  def vectorize(self, sentence: str) -> numpy.ndarray:
    """
    Return a tensor representation of the sentence of size (13 layers, num tokens, 768 dim).
    Even though there are only 12 layers in GPT2, we include the input embeddings as the first
    layer (for a fairer comparison to ELMo).
    """
    inputs = self.tokenizer.encode_plus(sentence, return_token_type_ids=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    # outputs = self.model(result.input_ids)
    # input_ids = self.tokenizer.encode(sentence, return_tensors="pt")
    # outputs = self.model(input_ids=input_ids, decoder_input_ids=input_ids, lm_labels=input_ids)
    # outputs = self.model.generate(**inputs)
    outputs = self.model(input_ids=input_ids, decoder_input_ids=input_ids, lm_labels=input_ids)
    output_embeddings = outputs[3]
    # embeddings = torch.stack(output_embeddings, dim=0).squeeze()[:,:,:]
    embeddings = torch.stack(output_embeddings, dim=0).squeeze()[:,:,:]
    embeddings = embeddings.detach().numpy()
    # print("NEW", embeddings.shape)            
    return embeddings

def words_from_sentence(s):
  return re.findall(r"[\w']+", s)

first_time = 0

def get_index_word_pairs_from_tokens(tokens, tokenizer):
  global first_time

  all_pairs = []

  # print(tokens)
  # print(tokenizer.decode(tokens))
  # TODO: remove pure digits
  words_in_sentence = words_from_sentence(tokenizer.decode(tokens))
  # words_in_sentence = words_from_sentence(tokenizer.decode(tokens))[1:-1]
  if first_time < 10:
    print(words_in_sentence)
    first_time = first_time + 1
  cur_target_word = 0
  cur_candate_token = 0
  for i in range(len(tokens)):
    candidates = words_from_sentence(tokenizer.decode(tokens[0:i+1]))
    # print(candidates)
    for j in range(cur_candate_token, len(candidates)):
      if cur_target_word >= len(words_in_sentence):
        pass
        # print("ignoring token {} because we're done".format(candidates[j]))
      elif words_in_sentence[cur_target_word] == candidates[j]:
        # if cur_candate_token < j:
        #   print("warning, skipping tokens:",candidates[cur_candate_token:j])
        # print("Found {} at position {}".format(words_in_sentence[cur_target_word], i))
        all_pairs.append([words_in_sentence[cur_target_word], i])
        cur_target_word = cur_target_word + 1
        cur_candate_token = j + 1
  return all_pairs

def index_tokens(tokens: List[str], tokenizer, sent_index: int, indexer: Dict[str, List[Tuple[int, int]]]) -> None:
  """
  Given string tokens that all appear in the same sentence, append tuple (sentence index, index of
  word in sentence) to the list of values each token is mapped to in indexer. Exclude tokens that 
  are punctuation.

  Args:
    tokens: string tokens that all appear in the same sentence
    sent_index: index of sentence in the data
    indexer: map of string tokens to a list of unique tuples, one for each sentence the token 
      appears in; each tuple is of the form (sentence index, index of token in that sentence)
  """
  word_pairs = get_index_word_pairs_from_tokens(tokens, tokenizer)
  for word, token_index in word_pairs:
    if word not in indexer:
      indexer[word] = []

    indexer[word].append((sent_index, token_index))


def index_sentence(data_fn: str, index_fn: str, tokenizer, min_count=5, do_lower_case=False, word_filter=None, sentence_prefix=None) -> List[str]:
  """
  Given a data file data_fn with the format of sts.csv, index the words by sentence in the order
  they appear in data_fn. 

  Args:
    index_fn: at index_fn, create a JSON file mapping each word to a list of tuples, each 
      containing the sentence it appears in and its index in that sentence
    tokenize: a callable function that maps each sentence to a list of string tokens; identity
      and number of tokens generated can vary across functions
    min_count: tokens appearing fewer than min_count times are left out of index_fn

  Return:
    List of sentences in the order they were indexed.
  """
  word2sent_indexer = {}
  sentences = []
  sentence_index = 0
  sentence_set = set()
  word_filter_set = set()

  # print(sentence_prefix)
  # print(tokenize, decoder, min_count, do_lower_case, word_filter)

  if word_filter is not None:
    # with open(word_filter) as f:
    #     content = f.readlines()
    # print(f"read {len(content)} lines from {word_filter}")
    # # you may also want to remove whitespace characters like `\n` at the end of each line
    # content = [x.strip() for x in content]
    # word_filter_set = set(content)
    print("opening {}".format(word_filter))
    with open(word_filter) as file_in:
      print("opened {}".format(word_filter))
      for line in file_in:
        word = line.strip()
        # print(f"Adding {word}")
        word_filter_set.add(word)
    # word_filter_set = set(["hello", "my", "friend"])

  print("Indexing {} -> {} with min_count {} and filter {}".format(data_fn, index_fn, min_count, len(word_filter_set)))
  minimum_word_count = 3
  maximum_word_count = 60
  if sentence_prefix is not None:
    print("Using sentence prefix: [{}]".format(sentence_prefix))
    minimum_word_count = minimum_word_count + len(words_from_sentence(sentence_prefix))

  duplicates_removed = 0
  sentences_filtered = 0
  if data_fn.endswith(".txt"):
    with open(data_fn) as fp:
       for cnt, line in enumerate(fp):
          sentence_candidate = line.strip()
          # print("Line {}: {}".format(cnt, line))
          if sentence_prefix is not None:
            sentence_candidate = f"{sentence_prefix}{sentence_candidate}"
          if sentence_candidate in sentence_set:
            # print("ignoring duplicate sentence: {}".format(sentence_candidate))
            duplicates_removed += 1
          else:
            # tokens = tokenizer.tokenize(sentence_candidate)
            words = words_from_sentence(sentence_candidate)
            # print(sentence_candidate, words)
            results = tokenizer.encode_plus(sentence_candidate, return_token_type_ids=True, return_tensors="pt")
            token_ids = results.input_ids[0].numpy()
            # print("checking", sentence_candidate, tokens)
            downcased_tokens = [w.lower() for w in words]
            if len(word_filter_set) > 0 and set(downcased_tokens).isdisjoint(word_filter_set):
              sentences_filtered += 1
              # print("skipping", tokens)
            elif len(words) < minimum_word_count:
              sentences_filtered += 1
              print("too short: {} -> {}".format(sentence_candidate, words))
            elif len(words) > maximum_word_count:
              sentences_filtered += 1
              print("too long: {} -> {}".format(sentence_candidate, words))
            else:
              # print("found", set(downcased_tokens).intersection(word_filter_set))
              index_tokens(token_ids, tokenizer, sentence_index, word2sent_indexer)
              sentences.append(sentence_candidate)
              sentence_set.add(sentence_candidate)
              sentence_index += 1
  else:
    with open(data_fn) as csvfile:
      csvreader = csv.DictReader(csvfile, quotechar='"', delimiter='\t')

      for line in csvreader:
        # only consider scored sentence pairs
        if line['Score'] == '':  
          continue

        # handle case where \t is between incomplete quotes (causes sents to be treated as one)
        if line['Sent2'] is None:
          line['Sent1'], line['Sent2'] = line['Sent1'].split('\t')[:2]

        for sentence_candidate in [line['Sent1'], line['Sent2']]:
          if do_lower_case:
            sentence_candidate = sentence_candidate.lower()
          if sentence_prefix is not None:
            sentence_candidate = f"{sentence_prefix}{sentence_candidate}"
          if sentence_candidate in sentence_set:
            # print("ignoring duplicate sentence: {}".format(sentence_candidate))
            duplicates_removed += 1
          else:
            # tokens = tokenizer.tokenize(sentence_candidate)
            words = words_from_sentence(sentence_candidate)
            # print(sentence_candidate, words)
            results = tokenizer.encode_plus(sentence_candidate, return_token_type_ids=True, return_tensors="pt")
            token_ids = results.input_ids[0].numpy()
            # print("checking", sentence_candidate, tokens)
            downcased_tokens = [w.lower() for w in words]
            if len(word_filter_set) > 0 and set(downcased_tokens).isdisjoint(word_filter_set):
              sentences_filtered += 1
              # print("skipping", tokens)
            elif len(words) < minimum_word_count:
              sentences_filtered += 1
              print("too short: {} -> {}".format(sentence_candidate, words))
            else:
              # print("found", set(downcased_tokens).intersection(word_filter_set))
              index_tokens(token_ids, tokenizer, sentence_index, word2sent_indexer)
              sentences.append(sentence_candidate)
              sentence_set.add(sentence_candidate)
              sentence_index += 1

  print("Sentence summary: found {} unique sentences ({} duplicates removed, {} filtered, lower_case={})".format(sentence_index, duplicates_removed, sentences_filtered, do_lower_case))
  # remove words that appear less than min_count times
  infrequent_words = list(filter(lambda w: len(word2sent_indexer[w]) < min_count, word2sent_indexer.keys()))
  
  for w in infrequent_words:
    del word2sent_indexer[w]

  json.dump(word2sent_indexer, open(index_fn, 'w'), indent=1)
  
  return sentences


# where to save the contextualized embeddings
EMBEDDINGS_PATH = "./contextual_embeddings"

def main():
    parser = argparse.ArgumentParser(description="pre process csv data file into hdf5")
    parser.add_argument('--suffix', default=None,
                         help='common suffix to all data files')
    parser.add_argument('--models', default="bert,gpt2,t5",
                         help='comma separated list of models to process')
    parser.add_argument('--alias', default=None,
                         help='one run model under a pseudonym')
    parser.add_argument('--input', default="inputs/sts.csv",
                         help='input file')
    parser.add_argument('--min-count', default=3, type=int,
                         help='minimum count threshold. less than this is cut')
    parser.add_argument('--do-lower-case', default=False, type=bool,
                         help='lowercase all input while reading')
    parser.add_argument('--word-filter', default=None,
                         help='keep only sentences that match set of words in filter')
    parser.add_argument('--model-name', default=None,
                         help='which t5 model to run')
    parser.add_argument('--t5-prefix', default='sst2 sentence: ',
                         help='which prefix (task) to run')
    args = parser.parse_args()

    models_to_process = args.models.split(",")

    file_suffix = ""
    if args.suffix is not None:
        file_suffix = "_{}".format(args.suffix)

    input_file = args.input

    if "bert" in models_to_process:
        if args.alias is None:
          outname = "bert"
        else:
          outname = args.alias

        if not os.path.exists(outname):
            os.makedirs(outname)

        index_file = '{}/word2sent{}.json'.format(outname, file_suffix)
        data_file = os.path.join(EMBEDDINGS_PATH, '{}{}.hdf5'.format(outname, file_suffix))

        model = Bert(args.model_name)
        sentences = index_sentence(input_file, index_file, model.tokenizer, args.min_count, args.do_lower_case, args.word_filter)
        model.make_hdf5_file(sentences, data_file)

    if "roberta" in models_to_process:
        index_file = 'roberta/word2sent{}.json'.format(file_suffix)
        data_file = os.path.join(EMBEDDINGS_PATH, 'roberta{}.hdf5'.format(file_suffix))

        model = Roberta()
        sentences = index_sentence(input_file, index_file, model.tokenizer, args.min_count, args.do_lower_case, args.word_filter)
        model.make_hdf5_file(sentences, data_file)

    if "t5" in models_to_process:
        t5_index_file = 't5/word2sent{}.json'.format(file_suffix)
        t5_data_file = os.path.join(EMBEDDINGS_PATH, 't5{}.hdf5'.format(file_suffix))

        t5 = T5Base(args.model_name, args.t5_prefix)
        sentences = index_sentence(input_file, t5_index_file, t5.tokenizer, args.min_count, args.do_lower_case, args.word_filter, t5.sentence_prefix)
        t5.make_hdf5_file(sentences, t5_data_file)

    if "gpt2" in models_to_process:
        gpt2_index_file = 'gpt2/word2sent{}.json'.format(file_suffix)
        gpt2_data_file = os.path.join(EMBEDDINGS_PATH, 'gpt2{}.hdf5'.format(file_suffix))

        gpt2 = GPT2()
        sentences = index_sentence(input_file, gpt2_index_file, gpt2.tokenizer, args.min_count, args.do_lower_case, args.word_filter)
        gpt2.make_hdf5_file(sentences, gpt2_data_file)

if __name__ == '__main__':
    main()
