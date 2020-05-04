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

matplotlib.rc('axes', edgecolor='k')

# where the contextualized embeddings are saved (in HDF5 format)
EMBEDDINGS_PATH = "./contextual_embeddings"

num_layers_table = {'t5': 13, 'newbert': 13, 'oldbert': 13, 'gpt2': 13, 'ELMo': 3}

def visualize_embedding_space(models_to_process, file_suffix):
	"""Plot the baseline charts in the paper. Images are written to the img/ subfolder."""
	plt.figure(figsize=(12,4))
	icons = [ 'ro:', 'bo:', 'go:']

	#for i, (model, num_layers) in enumerate([('ELMo', 3), ('BERT', 13), ('GPT2', 13)]):
	# for i, (model, num_layers) in enumerate([('bert', 13), ('gpt2', 13)]):
	for i, model in enumerate(models_to_process):
		num_layers = num_layers_table[model]
		x = np.array(range(num_layers))
		embedding_file = f'{model}/embedding_space_stats{file_suffix}.json'
		data = json.load(open(embedding_file))
		plt.plot(x, [ data["mean cosine similarity across words"][f'layer_{i}'] for i in x ], icons[i], markersize=6, label=model, linewidth=2.5, alpha=0.65)
		print(spearmanr(
			[ data["mean cosine similarity across words"][f'layer_{i}'] for i in x ],
			[ data["word norm std"][f'layer_{i}'] for i in x ]
		))

	plt.grid(True, linewidth=0.25)
	plt.legend(loc='upper left')
	plt.xlabel('Layer Index')
	plt.xticks(x)
	plt.ylim(0,1.0)
	plt.title("Average Cosine Similarity between Randomly Sampled Words")
	plt.savefig(f'img/mean_cosine_similarity_across_words{file_suffix}.png', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(12,4))
	icons = [ 'ro:', 'bo:', 'go:']

	#for i, (model, num_layers) in enumerate([('ELMo', 3), ('BERT', 13), ('GPT2', 13)]):
	# for i, (model, num_layers) in enumerate([('bert', 13), ('gpt2', 13)]):
	for i, model in enumerate(models_to_process):
		num_layers = num_layers_table[model]
		x = np.array(range(num_layers))
		embedding_file = f'{model}/embedding_space_stats{file_suffix}.json'
		data = json.load(open(embedding_file))
		y1 = np.array([ data["mean cosine similarity between sentence and words"][f'layer_{i}'] for i in x ])
		y2 = np.array([ data["mean cosine similarity across words"][f'layer_{i}'] for i in x ])
		plt.plot(x, y1 - y2, icons[i], markersize=6, label=model, linewidth=2.5, alpha=0.65)

	plt.grid(True, linewidth=0.25)
	plt.legend(loc='upper right')
	plt.xlabel('Layer Index')
	plt.xticks(x)
	plt.ylim(-0.1, 0.5)
	plt.title("Average Intra-Sentence Similarity (anisotropy-adjusted)")
	plt.savefig(f'img/mean_cosine_similarity_between_sentence_and_words{file_suffix}.png', bbox_inches='tight')
	plt.close()


def visualize_self_similarity(models_to_process, file_suffix):
	"""Plot charts relating to self-similarity. Images are written to the img/ subfolder."""
	plt.figure(figsize=(12,4))
	icons = [ 'ro:', 'bo:', 'go:']

	# plot the mean self-similarity but adjust by subtracting the avg similarity between random pairs of words
	#for i, (model, num_layers) in enumerate([('ELMo', 3), ('BERT', 13), ('GPT2', 13)]):
	# for i, (model, num_layers) in enumerate([('bert', 13), ('gpt2', 13)]):
	for i, model in enumerate(models_to_process):
		num_layers = num_layers_table[model]
		embedding_file = f'{model}/embedding_space_stats{file_suffix}.json'
		embedding_stats = json.load(open(embedding_file))
		self_similarity_file = f'{model}/self_similarity{file_suffix}.csv'
		self_similarity = pd.read_csv(self_similarity_file)

		x = np.array(range(num_layers))
		y1 = np.array([ self_similarity[f'layer_{i}'].mean() for i in x ])
		y2 = np.array([ embedding_stats["mean cosine similarity across words"][f'layer_{i}'] for i in x ])
		plt.plot(x, y1 - y2, icons[i], markersize=6, label=model, linewidth=2.5, alpha=0.65)

	plt.grid(True, linewidth=0.25)
	plt.legend(loc='upper right')
	plt.xlabel('Layer Index')
	plt.xticks(x)
	plt.ylim(0,1)
	plt.title("Average Self-Similarity (anisotropy-adjusted)")
	plt.savefig(f'img/self_similarity_above_expected{file_suffix}.png', bbox_inches='tight')
	plt.close()

	# list the top 10 words that are most self-similar and least self-similar 
	most_self_similar = []
	least_self_similar = []
	models = []

	#for i, (model, num_layers) in enumerate([('ELMo', 3), ('BERT', 13), ('GPT2', 13)]):
	# for i, (model, num_layers) in enumerate([('bert', 13), ('gpt2', 13)]):
	for i, model in enumerate(models_to_process):
		num_layers = num_layers_table[model]
		self_similarity_file = f'{model}/self_similarity{file_suffix}.csv'
		self_similarity = pd.read_csv(self_similarity_file)
		self_similarity['avg'] = self_similarity.mean(axis=1)

		models.append(model)
		most_self_similar.append(self_similarity.nlargest(10, 'avg')['word'].tolist())
		least_self_similar.append(self_similarity.nsmallest(10, 'avg')['word'].tolist())
	
	print(' & '.join(models) + '\\\\')
	for tup in zip(*most_self_similar): print(' & '.join(tup) + '\\\\')
	print()
	print(' & '.join(models) + '\\\\')
	for tup in zip(*least_self_similar): print(' & '.join(tup) + '\\\\')


def visualize_variance_explained(models_to_process, file_suffix):
	"""Plot chart for variance explained. Images are written to the img/ subfolder."""
	bar_width = 0.2
	plt.figure(figsize=(12,4))
	icons = [ 'ro:', 'bo:', 'go:']

	# plot the mean variance explained by first PC for occurrences of the same word in different sentences
	# adjust the values by subtracting the variance explained for random sentence vectors
	#for i, (model, num_layers) in enumerate([('ELMo', 3), ('BERT', 13), ('GPT2', 13)]):
	# for i, (model, num_layers) in enumerate([('bert', 13), ('gpt2', 13)]):
	for i, model in enumerate(models_to_process):
		num_layers = num_layers_table[model]
		embedding_file = f'{model}/embedding_space_stats{file_suffix}.json'
		embedding_stats = json.load(open(embedding_file))
		data = pd.read_csv(f'{model}/variance_explained{file_suffix}.csv')

		x = np.array(range(1, num_layers))
		y1 = np.array([ data[f'layer_{i}'].mean() for i in x ])
		y2 = np.array([ embedding_stats["variance explained for random words"][f'layer_{i}'] for i in x])
		plt.bar(x + i * bar_width, y1 - y2, bar_width, label=model, color=icons[i][0], alpha=0.65)

	plt.grid(True, linewidth=0.25, axis='y')
	plt.legend(loc='upper right')
	plt.xlabel('Layer')
	plt.xticks(x + i * bar_width / 2, x)
	plt.ylim(0,0.1)
	plt.axhline(y=0.05, linewidth=1, color='k', linestyle='--')
	plt.title("Average Maximum Explainable Variance (anisotropy-adjusted)")
	plt.savefig(f'img/variance_explained{file_suffix}.png', bbox_inches='tight')
	plt.show()
	plt.close()


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

    # paths to GloVe and FastText word vectors
    # http://nlp.stanford.edu/data/glove.42B.300d.zip
    # https://raw.githubusercontent.com/ekzhu/go-fasttext/master/testdata/wiki.en.vec
    # i mean https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
        #"~/csPMI/glove.42B.300d.txt",
        #"~/csPMI/wiki.en.vec"
    vector_paths = [
        "vector_paths/glove.42B.300d.txt",
        "vector_paths/wiki.en.vec"
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
    results = []
    # run the word_embedding_benchmarks code on the trimmed word vector files
    for fn in tqdm(os.listdir(trimmed_embedding_path)):
        pth = os.path.join(trimmed_embedding_path, fn)

        load_kwargs = {}
        file_length = sum(1 for line in open(pth))
        if file_length > 0:
            print("running benchmark on trimmed vector file {}, {}".format(fn, file_length))
            load_kwargs['vocab_size'] = file_length
            load_kwargs['dim'] = len(next(open(pth)).split()) - 1

            embeddings = load_embedding(pth, format='glove', normalize=True, lower=True,
                clean_words=False, load_kwargs=load_kwargs)
            df = evaluate_on_all(embeddings)
            df['Model'] = fn
            results.append(df)
        else:
            print("skipping empty file {}".format(pth))

    results = pd.concat(results).set_index('Model')
    return results


if __name__ == "__old_main__":
    visualize_embedding_space();
    visualize_self_similarity();
    visualize_variance_explained();
    results = evaluate();
    print(results);
    results.to_csv("results.csv", sep='\t')

def main():
    parser = argparse.ArgumentParser(description="pre process csv data file into hdf5")
    parser.add_argument('--suffix', default=None,
                         help='common suffix to all data files')
    parser.add_argument('--models', default="bert,gpt2",
                         help='comma separated list of models to process')
    parser.add_argument('--processes', default="similarity,variance,embedding,evaluate",
                         help='comma separated list of what to process')
    args = parser.parse_args()

    models_to_process = args.models.split(",")
    processes_to_run = args.processes.split(",")

    file_suffix = ""
    if args.suffix is not None:
        file_suffix = "_{}".format(args.suffix)

    if "similarity" in processes_to_run:
        print(f"Visualizing word similarity across sentences ...")
        visualize_self_similarity(models_to_process, file_suffix)

    if "variance" in processes_to_run:
        print(f"Visualizing variance explained by first principal component ...")
        visualize_variance_explained(models_to_process, file_suffix)

    if "embedding" in processes_to_run:
        print(f"Visualizing embedding space ...")
        visualize_embedding_space(models_to_process, file_suffix)

    if "evaluate" in processes_to_run:
        print(f"Evalutating ...")
        results = evaluate(models_to_process, file_suffix)
        results.to_csv("results/results{}.tsv".format(file_suffix), sep='\t')

if __name__ == '__main__':
    main()
