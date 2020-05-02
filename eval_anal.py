import os
import matplotlib
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from word_embeddings_benchmark.web.evaluate import evaluate_on_all
from word_embeddings_benchmark.web.embeddings import fetch_GloVe, load_embedding
import argparse
import shutil
from web.evaluate import evaluate_analogy
from web.analogy import fetch_google_analogy
from web.utils import standardize_string
from sklearn.datasets.base import Bunch
import numpy as np

matplotlib.rc('axes', edgecolor='k')

# where the contextualized embeddings are saved (in HDF5 format)
EMBEDDINGS_PATH = "./contextual_embeddings"

num_layers_table = {'bert': 13, 'gpt2': 13, 'ELMo': 3}

def fetch_all_analogies():
    with open("/home/tom/web_data/analogy/EN-GOOGLE/EN-GOOGLE.txt", "r") as f:
        L = f.read().splitlines()

    # Simple 4 word analogy questions with categories
    questions = []
    answers = []
    category = []
    cat = None
    for l in L:
        if l.startswith(":"):
            cat =l.lower().split()[1]
        else:
            words = standardize_string(l).split()
            questions.append(words[0:3])
            answers.append(words[3])
            category.append(cat)

    assert set(category) == set(['gram3-comparative', 'gram8-plural', 'capital-common-countries',
                                         'city-in-state', 'family', 'gram9-plural-verbs', 'gram2-opposite',
                                         'currency', 'gram4-superlative', 'gram6-nationality-adjective',
                                         'gram7-past-tense',
                                         'gram5-present-participle', 'capital-world', 'gram1-adjective-to-adverb'])


    syntactic = set([c for c in set(category) if c.startswith("gram")])
    category_high_level = []
    for cat in category:
         category_high_level.append("syntactic" if cat in syntactic else "semantic")

    # dtype=object for memory efficiency
    bunch = Bunch(X=np.vstack(questions).astype("object"),
                 y=np.hstack(answers).astype("object"),
                 category=np.hstack(category).astype("object"),
                 category_high_level=np.hstack(category_high_level).astype("object"))

    # return {"all": bunch}
    return bunch

wiki_6b_baseline = {
  "gram9-plural-verbs": 0.5850574712643678,
  "city-in-state": 0.5930279691933522,
  "gram4-superlative": 0.7219251336898396,
  "family": 0.8814229249011858,
  "gram3-comparative": 0.8813813813813813,
  "capital-common-countries": 0.9486166007905138,
  "gram6-nationality-adjective": 0.9255784865540964,
  "gram2-opposite": 0.2733990147783251,
  "currency": 0.1581986143187067,
  "gram5-present-participle": 0.6998106060606061,
  "gram1-adjective-to-adverb": 0.22580645161290322,
  "gram8-plural": 0.7807807807807807,
  "gram7-past-tense": 0.6115384615384616,
  "capital-world": 0.9597701149425287,
}

glove_42b_baseline = {
  "gram4-superlative": 0.8368983957219251,
  "currency": 0.1697459584295612,
  "gram3-comparative": 0.8558558558558559,
  "family": 0.9090909090909091,
  "gram8-plural": 0.8490990990990991,
  "capital-world": 0.9383289124668435,
  "gram1-adjective-to-adverb": 0.3024193548387097,
  "gram9-plural-verbs": 0.6413793103448275,
  "gram2-opposite": 0.35591133004926107,
  "gram7-past-tense": 0.492948717948718,
  "gram5-present-participle": 0.8087121212121212,
  "city-in-state": 0.7807053100932306,
  "capital-common-countries": 0.950592885375494,
  "gram6-nationality-adjective": 0.8830519074421513,
}

def evaluate_analogies(embed_name, w):
    data = fetch_all_analogies()
    all_categories = set(data["category"])

    catogory_results = []
    weighted_result = []
    num_results = []
    result_table = {"name": embed_name}
    for cur_category in all_categories:
      is_cat = (data["category"] == cur_category)
      # cat_table = np.extract(is_cat, datatable["category"])
      cur_X = data.X[is_cat]
      cur_y = data.y[is_cat]
      # print(cur_category, len(is_cat), data.X.shape, data.y.shape, cur_X.shape, cur_y.shape)
      result = evaluate_analogy(w, cur_X, cur_y)
      scaled_result = result / glove_42b_baseline[cur_category]
      # print(cur_category, len(cur_y), result, scaled_result)
      # print(embed_name, cur_category, result, scaled_result)
      catogory_results.append(scaled_result)
      num_results.append(len(cur_y))
      weighted_result.append(len(cur_y) * result)
      result_table[cur_category] = result
      result_table[f"{cur_category}_name"] = embed_name
      result_table[f"{cur_category}_weight"] = len(cur_y)

    result2 = evaluate_analogy(w, data.X, data.y)

    net_result = np.sum(weighted_result) / np.sum(num_results)
    result_table["net_score"] = net_result
    result_table["total_score"] = np.mean(catogory_results)
    result_table["raw_score"] = result2
    return result_table

    # print(embed_name, "Raw Score:", raw_result)
    # print(embed_name, "Total Score:", np.mean(catogory_results))
    # print(embed_name, "Net Score:", result2)

def report_analogy_results(result_table, baseline_table):
    embed_name = result_table["name"]
    print("====== ", embed_name)
    weighted_result = []
    num_results = []
    for k in sorted(glove_42b_baseline.keys()):
      result = result_table[k]
      baseline_result = baseline_table[k]
      report_name = result_table[f"{k}_name"]
      cur_num_results = result_table[f"{k}_weight"]
      num_results.append(cur_num_results)
      weighted_result.append(cur_num_results * result)
      # print(k, baseline_result, type(baseline_result))
      # if (type(baseline_result) == 'int' or type(baseline_result) == 'float') and baseline_result > 0:
      #   scaled_result = result / baseline_result
      # else:
      #   scaled_result = -1
      if baseline_result != 0:
        scaled_result = result / baseline_result
      else:
        if result > 0:
            scaled_result = 10.0
        else:
            scaled_result = 0
      print(report_name, k, result, scaled_result)

    net_result = np.sum(weighted_result) / np.sum(num_results)
    print("====== ", embed_name, " NET RESULT ", net_result)

    for k in ["net_score", "raw_score", "total_score"]:
      if k in result_table:
        result = result_table[k]
        baseline_result = baseline_table[k]
        scaled_result = result / baseline_result
        print(embed_name, k, result, scaled_result)

def evaluate(models_to_process, file_suffix):
    """
    Evaluate both typical word embeddings (GloVe, FastText) and word embeddings created by taking the
    first PC of contextualized embeddings on standard benchmarks for word vectors (see paper for
    details). These benchmarks include tasks like arithmetic analogy-solving. Paths in this function
    are hard-coded and should be modified as needed.

    First, create a smaller version of the embedding files where vocabulary contains only the words
    that have some ELMo vector. Then, run the code in the word_embeddings_benchmarks library on the
    trimmed word vector files.

    Returns:
        A DataFrame where the index is the model layer from which the PC vectors are derived and
        each column contains the performance on a particular task. See the word_embeddings_benchmarks
        library for details on these tasks.
    """


    # create a smaller version of the embedding files where vocabulary = only words that have some ELMo vector
    # vocabulary needs to be the same across all embeddings for apples-to-apples comparison
    words = None
    for model in models_to_process:
		# words = set([ w.lower() for w in json.load(open('bert/word2sent.json')).keys() ])

        word2sent_file = f'{model}/word2sent{file_suffix}.json'
        new_words = set([ w.lower() for w in json.load(open(word2sent_file)).keys() ])
        print("found {} words in {}".format(len(new_words), model))
        if words is None:
            words = new_words
        else:
            words = words.intersection(new_words)
    print("kept {} words in vocabulary".format(len(words)))

    with open(f"results/words{file_suffix}.txt", 'w') as f:
        f.write("\n".join(map(str, words)))

    # paths to GloVe and FastText word vectors
    # http://nlp.stanford.edu/data/glove.42B.300d.zip
    # https://raw.githubusercontent.com/ekzhu/go-fasttext/master/testdata/wiki.en.vec
    # i mean https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
        #"~/csPMI/glove.42B.300d.txt",
        #"~/csPMI/wiki.en.vec"
        # "vector_paths/wiki.en.vec"
    vector_paths = [
        "vector_paths/glove.42B.300d.txt",
    ]

    # paths to the principal components of the contextualized embeddings
    # each layer of each model (ELMo, BERT, GPT2) should have its own set
    pc_path = "./contextual_embeddings/pcs"

    # paths to ELMo embeddings
    #for i in range(1,3):
    #    vector_paths.append(os.path.join(pc_path, f'elmo.pc.{i}'))

    # paths to BERT and GPT2 embeddings
    for model in models_to_process:
        for i in range(1,13):
            vector_paths.append(os.path.join(EMBEDDINGS_PATH, f'pcs/{model}{file_suffix}.pc.{i}'))

    if file_suffix is None or len(file_suffix) == 0:
        path_leaf = "none"
    elif file_suffix[0] == '_':
        path_leaf = file_suffix[1:]
    else:
        path_leaf = file_suffix

    # where to put the smaller embedding files
    trimmed_embedding_path = f'{EMBEDDINGS_PATH}/trimmed/{path_leaf}'
    if os.path.exists(trimmed_embedding_path):
        print("cleaning out previous contents of {}".format(trimmed_embedding_path))
        shutil.rmtree(trimmed_embedding_path)
    print("Creating trimmed results in {}".format(trimmed_embedding_path))
    os.makedirs(trimmed_embedding_path)

    num_found = 0
    for path in tqdm(vector_paths):
        name = path.split('/')[-1]

        with open(os.path.join(trimmed_embedding_path, name), 'w') as f_out:
            for line in open(path):
                if line.split()[0].lower() in words:
                    num_found = num_found + 1
                    f_out.write(line.strip() + '\n')

    print("vocabulary found {} words".format(num_found))
    result_table = {}
    # run the word_embedding_benchmarks code on the trimmed word vector files
    file_list = sorted(os.listdir(trimmed_embedding_path))
    for fn in tqdm(file_list):
        pth = os.path.join(trimmed_embedding_path, fn)

        load_kwargs = {}
        file_length = sum(1 for line in open(pth))
        if file_length > 0:
            print("running benchmark on trimmed vector file {}, {}".format(fn, file_length))
            load_kwargs['vocab_size'] = file_length
            load_kwargs['dim'] = len(next(open(pth)).split()) - 1

            embeddings = load_embedding(pth, format='glove', normalize=True, lower=True,
                clean_words=False, load_kwargs=load_kwargs)
            result_table[fn] = evaluate_analogies(fn, embeddings)
            # df['Model'] = fn
            # results.append(df)
        else:
            print("skipping empty file {}".format(pth))

    best_overall = None
    best_overall_score = 0
    best_each_section = {"name": "mixed"}
    best_each_section_score = {}
    for fn in file_list:
        if fn[:5] != 'glove':
            if best_overall is None or (best_overall_score < result_table[fn]["total_score"]):
                best_overall = fn
                best_overall_score = result_table[fn]["total_score"]
            for k in glove_42b_baseline.keys():
                if not k in best_each_section or (best_each_section_score[k] < result_table[fn][k]):
                    best_each_section[k] = result_table[fn][k]
                    best_each_section[f"{k}_name"] = result_table[fn][f"{k}_name"]
                    best_each_section[f"{k}_weight"] = result_table[fn][f"{k}_weight"]
                    best_each_section_score[k] = result_table[fn][k]

        # report_analogy_results(result_table[fn], result_table["glove.42B.300d.txt"])

    print("===== BEST LAYER")
    # print(result_table[best_overall])
    report_analogy_results(result_table[best_overall], result_table["glove.42B.300d.txt"])
    print("===== BEST MIXED")
    # print(best_each_section)
    # report_analogy_results(best_each_section, result_table["glove.42B.300d.txt"])
    report_analogy_results(best_each_section, result_table[best_overall])

    # results = pd.concat(results).set_index('Model')
    # return results
    return None

def main():
    parser = argparse.ArgumentParser(description="pre process csv data file into hdf5")
    parser.add_argument('--suffix', default=None,
                         help='common suffix to all data files')
    parser.add_argument('--models', default="bert,gpt2",
                         help='comma separated list of models to process')
    args = parser.parse_args()

    models_to_process = args.models.split(",")

    file_suffix = ""
    if args.suffix is not None:
        file_suffix = "_{}".format(args.suffix)

    print(f"Evalutating ...")
    results = evaluate(models_to_process, file_suffix)
    # results.to_csv("results/anal_results{}.tsv".format(file_suffix), sep='\t')

if __name__ == '__main__':
    main()
