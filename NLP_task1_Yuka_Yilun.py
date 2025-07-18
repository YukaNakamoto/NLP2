import os
import codecs
import random
import argparse
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt 


# Task 1: Data Exploration

# Reads EN and ES files, returns list of (en, es) pairs.
def read_parallel_corpus(en_path, es_path, encoding='utf-8'):

    pairs = []                                                           # container for sentence pairs
    with codecs.open(en_path, 'r', encoding=encoding, errors='ignore') as f_en, \
         codecs.open(es_path, 'r', encoding=encoding, errors='ignore') as f_es:
        for en_line, es_line in zip(f_en, f_es):                        # read line-by-line in parallel
            en = en_line.strip()                            # remove whitespace/newlines
            es = es_line.strip()
            
            if not en or not es or en.startswith('<') or es.startswith('<'):  # skip empty lines or metadata tags
                continue
            pairs.append((en, es))     # add valid pair
    return pairs                 # return list of sentence pairs


# Compute and print mean/median/min/max for EN & ES lengths.
def summarize_lengths(en_lengths, es_lengths):
    def stats(lst):
        arr = np.array(lst)                                     # convert to numpy array
        return arr.mean(), np.median(arr), arr.min(), arr.max()

    en_stats = stats(en_lengths)                                  # stats for English lengths
    es_stats = stats(es_lengths)                                # stats for Spanish lengths
    ratio = [e / s if s > 0 else 0 for e, s in zip(en_lengths, es_lengths)]  # EN/ES length ratios
    ratio_stats = stats(ratio)

    print(f"English tokens: mean={en_stats[0]:.2f}, median={en_stats[1]:.2f}, min={en_stats[2]}, max={en_stats[3]}")
    print(f"Spanish tokens: mean={es_stats[0]:.2f}, median={es_stats[1]:.2f}, min={es_stats[2]}, max={es_stats[3]}")
    print(f"Length ratio (EN/ES): mean={ratio_stats[0]:.2f}, median={ratio_stats[1]:.2f}, min={ratio_stats[2]:.2f}, max={ratio_stats[3]:.2f}")


def plot_distributions(en_lengths, es_lengths, out_dir):
    os.makedirs(out_dir, exist_ok=True)                     # create output directory
    # plot histogram of token lengths
    plt.figure(figsize=(10,4))                             # set figure size
    plt.hist(en_lengths, bins=50, alpha=0.6, label='English')  # EN histogram
    plt.hist(es_lengths, bins=50, alpha=0.6, label='Spanish')  # ES histogram
    plt.title('Sentence length distribution (tokens)')    # add title
    plt.xlabel('Tokens per sentence')                  # x-axis label
    plt.ylabel('Count')                              # y-axis label
    plt.legend()                                        # show legend
    hist_path = os.path.join(out_dir, 'length_distribution.png')  # output path
    plt.savefig(hist_path)                        # save figure
    print(f"Saved histogram to {hist_path}")
    plt.close()                             # close plot

    # plot scatter of EN vs ES lengths
    plt.figure(figsize=(6,6))
    plt.scatter(en_lengths, es_lengths, s=1, alpha=0.3)  # small dots for clarity
    plt.title('EN vs ES sentence lengths')
    plt.xlabel('English tokens')
    plt.ylabel('Spanish tokens')
    scatter_path = os.path.join(out_dir, 'length_scatter.png')
    plt.savefig(scatter_path)
    print(f"Saved scatter plot to {scatter_path}")
    plt.close()


def get_top_tokens(sentences, n=20):
    tokens = [tok for sent in sentences for tok in sent.split()]  # flatten token list
    return Counter(tokens).most_common(n)                          # return top-n frequent tokens


def task1(args):
    print("Reading corpus...")
    data = read_parallel_corpus(args.en_file, args.es_file, encoding=args.encoding)      # load data
    total = len(data)                          # total number of pairs
    print(f"Total sentence pairs: {total}")

    # compute sentence lengths in tokens
    en_lengths = [len(e.split()) for e, _ in data]
    es_lengths = [len(s.split()) for _, s in data]

    # display summary statistics
    summarize_lengths(en_lengths, es_lengths)

    # calculate percentage of pairs under various length thresholds
    thresholds = [30, 50, 70, 100]
    for th in thresholds:
        count = sum(1 for e_len, s_len in zip(en_lengths, es_lengths) if e_len <= th and s_len <= th)  # count valid pairs
        pct = count / total * 100
        print(f"Pairs with both EN & ES <= {th} tokens: {count} ({pct:.2f}%)")

    # generate and save plots
    out_dir = 'task1_outputs'
    plot_distributions(en_lengths, es_lengths, out_dir)

    # compute and display top tokens
    en_sentences = [e for e, _ in data]
    es_sentences = [s for _, s in data]
    print("Top 20 English tokens:", get_top_tokens(en_sentences))
    print("Top 20 Spanish tokens:", get_top_tokens(es_sentences))

    # randomly sample 10% of data for downstream tasks
    sample_size = int(total * args.sample_ratio)         # compute sample size
    sample = random.sample(data, sample_size)            # random sample selection
    sample_en = os.path.join(out_dir, 'sample_10pct.en')  # EN output file
    sample_es = os.path.join(out_dir, 'sample_10pct.es')  # ES output file
    with open(sample_en, 'w', encoding='utf-8') as fe, open(sample_es, 'w', encoding='utf-8') as fs:
        for e, s in sample:
            fe.write(e + '\n')                       # write English sentences
            fs.write(s + '\n')                       # write Spanish sentences
    print(f"Saved 10% sample: {sample_en}, {sample_es}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 1: Data Exploration for EN-ES corpus')  # parser setup
    parser.add_argument('--en_file', type=str, default='europarl-v7.es-en.en', help='Path to English corpus')  # EN path
    parser.add_argument('--es_file', type=str, default='europarl-v7.es-en.es', help='Path to Spanish corpus')  # ES path
    parser.add_argument('--encoding', type=str, default='utf-8', help='File encoding (e.g., latin-1)')  # file encoding
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='Sampling ratio for Task 1')  # sample ratio
    args = parser.parse_args()                           # parse args

    task1(args)
