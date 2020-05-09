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
import pytorch_pretrained
from pytorch_pretrained import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import transformers
# from allennlp.commands.elmo import ElmoEmbedder
# from allennlp.data.tokenizers.token import Token
# from allennlp.common.tqdm import Tqdm
import tqdm

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

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

class ELMo(Vectorizer):
  def __init__(self):
    self.elmo = ElmoEmbedder()

  def vectorize(self, sentence: str) -> numpy.ndarray:
    """
    Return a tensor representation of the sentence of size (3 layers, num tokens, 1024 dim).
    """
    # tokenizer's tokens must be converted to string tokens first
    tokens = list(map(str, spacy_tokenizer(sentence)))  
    embeddings = self.elmo.embed_sentence(tokens)
    return embeddings   


class OldBertBaseCased(Vectorizer):
  def __init__(self):
    self.tokenizer = pytorch_pretrained.BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    self.model = pytorch_pretrained.BertModel.from_pretrained('bert-base-cased')
    self.model.eval()

  def vectorize(self, sentence: str) -> numpy.ndarray:
    """
    Return a tensor representation of the sentence of size (13 layers, num tokens, 768 dim).
    Even though there are only 12 layers in GPT2, we include the input embeddings as the first
    layer (for a fairer comparison to ELMo).
    """
    # add CLS and SEP to mark the start and end
    tokens = ['[CLS]'] + self.tokenizer.tokenize(sentence) + ['[SEP]']
    # tokenize sentence with custom BERT tokenizer
    token_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
    # segment ids are all the same (since it's all one sentence)
    segment_ids = numpy.zeros_like(token_ids)

    tokens_tensor = torch.tensor([token_ids])
    segments_tensor = torch.tensor([segment_ids])

    with torch.no_grad():
      embeddings, _, input_embeddings = self.model(tokens_tensor, segments_tensor)

    # exclude embeddings for CLS and SEP; then, convert to numpy
    embeddings = torch.stack([input_embeddings] + embeddings, dim=0).squeeze()[:,1:-1,:]
    embeddings = embeddings.detach().numpy()
    # print("OLD", embeddings.shape)            

    return embeddings


class NewBertBaseCased(Vectorizer):
  def __init__(self):
    self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
    config = transformers.BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
    self.model = transformers.BertModel.from_pretrained("bert-base-cased", config=config)

  def vectorize(self, sentence: str) -> numpy.ndarray:
    """
    Return a tensor representation of the sentence of size (13 layers, num tokens, 768 dim).
    Even though there are only 12 layers in GPT2, we include the input embeddings as the first
    layer (for a fairer comparison to ELMo).
    """
    tokens = self.tokenizer.encode(sentence, add_special_tokens=True)
    input_ids = torch.tensor(tokens).unsqueeze(0)  # Batch size 1
    outputs = self.model(input_ids)
    output_embeddings = outputs[2]
    embeddings = torch.stack(output_embeddings, dim=0).squeeze()[:,1:-1,:]
    embeddings = embeddings.detach().numpy()
    # print("NEW", embeddings.shape)            
    return embeddings

class T5Base(Vectorizer):
  def __init__(self):
    self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
    config = T5Config.from_pretrained("t5-base", output_hidden_states=True)
    self.model = T5ForConditionalGeneration.from_pretrained('t5-base', config=config)

  def vectorize(self, sentence: str) -> numpy.ndarray:
    """
    Return a tensor representation of the sentence of size (13 layers, num tokens, 768 dim).
    Even though there are only 12 layers in GPT2, we include the input embeddings as the first
    layer (for a fairer comparison to ELMo).
    """
    input_ids = self.tokenizer.encode(sentence, return_tensors="pt")
    outputs = self.model(input_ids=input_ids, decoder_input_ids=input_ids, lm_labels=input_ids)
    output_embeddings = outputs[3]
    embeddings = torch.stack(output_embeddings, dim=0).squeeze()[:,:,:]
    embeddings = embeddings.detach().numpy()
    # print("NEW", embeddings.shape)            
    return embeddings


class GPT2(Vectorizer):
  def __init__(self):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.model = GPT2Model.from_pretrained('gpt2')
    self.model.eval()

  def vectorize(self, sentence: str) -> numpy.ndarray:
    """
    Return a tensor representation of the sentence of size (13 layers, num tokens, 768 dim).
    Even though there are only 12 layers in GPT2, we include the input embeddings (with 
    positional information).

    For an apples-to-apples comparison with ELMo, we use the spaCy tokenizer at first -- to 
    avoid breaking up a word into subwords if possible -- and then fall back to using the GPT2 
    tokenizer if needed.
    """
    # use spacy tokenizer at first
    tokens_tentative = list(map(str, spacy_tokenizer(sentence)))
    token_ids_tentative = self.tokenizer.convert_tokens_to_ids(tokens_tentative)

    tokens = []
    token_ids = []

    for i, tid in enumerate(token_ids_tentative):
      # if not "unknown token" ID = 0, proceed
      if tid != 0:
        tokens.append(tokens_tentative[i])
        token_ids.append(tid)
      else:
        # otherwise, try to find a non-zero token ID by preprending the special character
        special_char_tid = self.tokenizer.convert_tokens_to_ids('Ġ' + tokens_tentative[i])
        # otherwise, break up the word using the given GPT2 tokenizer
        subtoken_tids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tokens_tentative[i]))

        if special_char_tid != 0:
          tokens.append('Ġ' + tokens_tentative[i])
          token_ids.append(special_char_tid)
        else:
          tokens.extend(self.tokenizer.tokenize(tokens_tentative[i]))
          token_ids.extend(subtoken_tids)

    tokens_tensor = torch.tensor([token_ids])

    # predict hidden states features for each layer
    with torch.no_grad():
        _, _, embeddings = self.model(tokens_tensor)

    embeddings = torch.stack(embeddings, dim=0).squeeze().detach().numpy()              
    
    return embeddings  


def index_tokens(tokens: List[str], sent_index: int, indexer: Dict[str, List[Tuple[int, int]]]) -> None:
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
  for token_index, token in enumerate(tokens):
    if not nlp.vocab[token].is_punct:
      if str(token) not in indexer:
        indexer[str(token)] = []

      indexer[str(token)].append((sent_index, token_index))


def index_sentence(data_fn: str, index_fn: str, tokenize: Callable[[str], List[str]], min_count=5, do_lower_case=False, word_filter=None) -> List[str]:
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

  with open(data_fn) as csvfile:
    csvreader = csv.DictReader(csvfile, quotechar='"', delimiter='\t')

    duplicates_removed = 0
    sentences_filtered = 0
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
        if sentence_candidate in sentence_set:
          # print("ignoring duplicate sentence: {}".format(sentence_candidate))
          duplicates_removed += 1
        else:
          tokens = tokenize(sentence_candidate)
          # print("checking", sentence_candidate, tokens)
          downcased_tokens = [w.lower() for w in tokens]
          if len(word_filter_set) > 0 and set(downcased_tokens).isdisjoint(word_filter_set):
            sentences_filtered += 1
            # print("skipping", tokens)
          elif len(tokens) < 3:
            sentences_filtered += 1
            print("too short: {} -> {}".format(sentence_candidate, tokens))
          else:
            # print("found", set(downcased_tokens).intersection(word_filter_set))
            index_tokens(tokens, sentence_index, word2sent_indexer)
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
    parser.add_argument('--models', default="bert,gpt2",
                         help='comma separated list of models to process')
    parser.add_argument('--input', default="inputs/sts.csv",
                         help='input file')
    parser.add_argument('--min-count', default=3, type=int,
                         help='minimum count threshold. less than this is cut')
    parser.add_argument('--do-lower-case', default=False, type=bool,
                         help='lowercase all input while reading')
    parser.add_argument('--word-filter', default=None,
                         help='keep only sentences that match set of words in filter')
    args = parser.parse_args()

    models_to_process = args.models.split(",")

    file_suffix = ""
    if args.suffix is not None:
        file_suffix = "_{}".format(args.suffix)

    input_file = args.input

    if "bert" in models_to_process:
        newbert_index_file = 'bert/word2sent{}.json'.format(file_suffix)
        newbert_data_file = os.path.join(EMBEDDINGS_PATH, 'bert{}.hdf5'.format(file_suffix))

        newbert = NewBertBaseCased()
        sentences = index_sentence(input_file, newbert_index_file, newbert.tokenizer.tokenize, args.min_count, args.do_lower_case, args.word_filter)
        newbert.make_hdf5_file(sentences, newbert_data_file)

    if "t5" in models_to_process:
        t5_index_file = 't5/word2sent{}.json'.format(file_suffix)
        t5_data_file = os.path.join(EMBEDDINGS_PATH, 't5{}.hdf5'.format(file_suffix))

        t5 = T5Base()
        sentences = index_sentence(input_file, t5_index_file, t5.tokenizer.tokenize, args.min_count, args.do_lower_case, args.word_filter)
        t5.make_hdf5_file(sentences, t5_data_file)

    if "oldbert" in models_to_process:
        oldbert_index_file = 'oldbert/word2sent{}.json'.format(file_suffix)
        oldbert_data_file = os.path.join(EMBEDDINGS_PATH, 'oldbert{}.hdf5'.format(file_suffix))

        oldbert = OldBertBaseCased()
        sentences = index_sentence(input_file, oldbert_index_file, oldbert.tokenizer.tokenize, args.min_count, args.do_lower_case, args.word_filter)
        oldbert.make_hdf5_file(sentences, oldbert_data_file)

    if "gpt2" in models_to_process:
        gpt2_index_file = 'gpt2/word2sent{}.json'.format(file_suffix)
        gpt2_data_file = os.path.join(EMBEDDINGS_PATH, 'gpt2{}.hdf5'.format(file_suffix))

        gpt2 = GPT2()
        sentences = index_sentence(input_file, gpt2_index_file, lambda s: list(map(str, spacy_tokenizer(s))), args.min_count, args.do_lower_case, args.word_filter)
        gpt2.make_hdf5_file(sentences, gpt2_data_file)

if __name__ == '__main__':
    main()
