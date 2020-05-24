#!/usr/bin/env python3

import argparse
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from collections import defaultdict
from scipy.sparse import csr_matrix
# from scipy.sparse import save_npz
# from nltk.stem import LancasterStemmer
# from nltk.corpus import stopwords
import numpy as np
import torch
import time
import heapq


# Lex must be sorted
def bin_search(term, lex, lo=0, hi=-1):
    if hi == -1:
        hi = len(lex)

    if lo == hi:
        raise Exception("Bin search failed")

    mid = lo + (hi - lo) // 2
    if lex[mid] == term:
        return mid

    if term < lex[mid]:
        return bin_search(term, lex, lo=lo, hi=mid)

    return bin_search(term, lex, lo=mid + 1, hi=hi)


def bm25_topk(pdoc, pdoc_id, docs, k, bsize=100):
    num_iter = int(np.ceil(docs.shape[0] / bsize))
    pdoc = torch.from_numpy(pdoc.todense()).double()
    scores = None
    for i in range(num_iter):
        batch = docs[i * bsize:(i + 1) * bsize].todense()
        batch = torch.from_numpy(batch).double()
        pmask = pdoc > 0
        pmask = pmask.repeat(batch.shape[0], 1)
        i_scores = torch.sum(pmask * batch, dim=1)
        if scores is None:
            scores = i_scores
        else:
            scores = torch.cat((scores, i_scores))

    _, topk_i = torch.topk(scores, k + 1)
    res_topk = []
    piv_seen = False
    for di in topk_i.tolist():
        if di != pdoc_id:
            res_topk.append(di)
        else:
            piv_seen = True

    if not piv_seen:
        raise Exception("Document itself must be in topk")

    return torch.tensor(res_topk)


def knn_all_docs(docs, k=20):
    knn = None
    start = time.time()
    for i, doc in enumerate(docs):
        doc_topk = bm25_topk(doc, i, docs, k)
        if knn is None:
            knn = doc_topk
        else:
            knn = torch.cat((knn, doc_topk))

        end = time.time()
        print(f"Computed knn for {i + 1}'th doc in {end-start}s")
        start = end

    return knn


def ir_topk(query, inv_index, doc_lens, k):
    doc_scores = defaultdict(float)
    avg_doclen = np.mean(doc_lens)
    k1 = 1.6
    b = 0.75
    for t in query:
        plist = inv_index[t]
        df = len(plist)
        idf = np.log2((len(inv_index) - df + 0.5) / df + 0.5)
        for did, tf in inv_index[t]:
            score = tf * (k1 + 1)
            score /= tf + k1 * (1 - b + b * doc_lens[did] / avg_doclen)
            score *= idf
            doc_scores[did] += score
    res = heapq.nlargest(k, [(score, did) for did, score in doc_scores.items()])
    return [did for score, did in res]


def ir_bm25_knn(docs, inv_index, doc_lens, k=20):
    knn = None
    start = time.time()
    for i, d in enumerate(docs):
        dres = ir_topk(d, inv_index, doc_lens, k + 1)
        dres.remove(i)
        dres = torch.tensor(dres)
        if knn is None:
            knn = dres
        else:
            knn = torch.cat((knn, dres))
        end = time.time()
        print(f"Computed knn for {i + 1}'th doc in {end-start}s")

    return knn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="ng20.npz",
                        help="Output file")
    parser.add_argument("-i", "--input", help="Input ng20 docs")
    args = parser.parse_args()
    # newsgroups = fetch_20newsgroups(remove=("headers", "footers", "quotes"),
    #                                 subset="all")
    # vectorizer = TfidfVectorizer()
    # preprocessor = vectorizer.build_preprocessor()
    # tokenizer = vectorizer.build_tokenizer()
    # raw_docs = [tokenizer(preprocessor(d)) for d in newsgroups["data"]]
    # stemmer = LancasterStemmer()
    # stop_words = set(stopwords.words("english"))
    docs = []
    categories = []
    # for rd in raw_docs:
    #     new_doc = []
    #     for w in rd:
    #         stw = stemmer.stem(w)
    #         if stw not in stop_words:
    #             new_doc.append(stw)
    #     docs.append(new_doc)
    with open(args.input) as ng20s:
        lines = ng20s.readlines()

    try:
        for i, l in enumerate(lines):
            cat, doc = l.rstrip().split("\t", maxsplit=1)
            if doc:
                docs.append(doc.split())
            else:
                raise Exception(f"Empty doc? line: {i + 1}")

            if cat:
                categories.append(cat)
            else:
                raise Exception(f"Empty cat? line: {i + 1}")

    except Exception:
        print(f"Line with problem {i + 1}")
        return 1
    # raw_docs = [[stemmer.stem(w) for w in doc if w not in stop_words] for doc
    #             in raw_docs]
    categories = np.array(categories).reshape(len(categories), 1)
    doc_lens = []
    lexicon = set()
    inv_index = defaultdict(list)
    docs_w_counts = []
    for i, rd in enumerate(docs):
        doc_lens.append(len(rd))
        wrd_count = defaultdict(int)
        for w in rd:
            wrd_count[w] += 1
            lexicon.add(w)

        for word, count in wrd_count.items():
            inv_index[word].append((i, count))

        docs_w_counts.append(wrd_count)

    doc_lens = np.array(doc_lens)
    print(f"lex size: {len(lexicon)}")
    lexicon = sorted(list(lexicon))
    lex_len = len(lexicon)
    idfs = []
    print("Computing idfs")
    for w in lexicon:
        df = len(inv_index[w])
        idfs.append(np.log2((lex_len - df + 0.5) / df + 0.5))

    idfs = np.array(idfs)
    print("Computing term freqs")
    freq_row = []
    freq_col = []
    data = []
    for i, dcount in enumerate(docs_w_counts):
        for w, f in dcount.items():
            freq_row.append(i)
            freq_col.append(bin_search(w, lexicon))
            data.append(f)

    term_freqs = csr_matrix((data, (freq_row, freq_col)),
                            shape=(len(doc_lens), lex_len), dtype=float)

    print("Computing BM25 weights")
    k1 = 1.6
    b = 0.75
    bm25_nom = term_freqs.multiply(idfs)
    bm25_nom = bm25_nom.multiply(k1 + 1)
    bm25_denom = k1 * (1 - b + b * doc_lens / np.mean(doc_lens))
    bm25_denom = bm25_denom.reshape((len(doc_lens), 1))
    bm25_denom = term_freqs._add_sparse(bm25_denom)
    np.reciprocal(bm25_denom.data, out=bm25_denom.data)
    bm25 = bm25_nom.multiply(bm25_denom)
    normalize(bm25, copy=False)
    # print("Computing KNN")
    # knn_data = ir_bm25_knn(docs, inv_index, doc_lens)
    print(f"Saving data to {args.output}")
    index = np.arange(bm25.shape[0])
    np.random.shuffle(index)
    np.savez_compressed(args.output, docs=bm25[index],
                        categories=categories[index])
    return 0


if __name__ == "__main__":
    exit(main())
