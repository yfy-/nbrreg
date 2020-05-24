#!/usr/bin/env python3

import argparse
import torch
import numpy as np
import torch.distributions as tdist
import scipy.io


class NbrReg(torch.nn.Module):
    def __init__(self, lex_size, bit_size, h_size=1000):
        super(NbrReg, self).__init__()
        self.lnr_h1 = torch.nn.Linear(lex_size, h_size)
        self.lnr_h2 = torch.nn.Linear(h_size, h_size)
        self.lnr_mu = torch.nn.Linear(h_size, bit_size)
        self.lnr_sigma = torch.nn.Linear(h_size, bit_size)
        self.lnr_rec_doc = torch.nn.Linear(bit_size, lex_size)
        self.lnr_nn_rec_doc = torch.nn.Linear(bit_size, lex_size)

    def forward(self, docs):
        mu, sigma = self.encode(docs)
        qdist = tdist.Normal(mu, sigma)
        log_prob_words, log_nn_prob_words = self.decode(qdist.rsample())
        return qdist, log_prob_words, log_nn_prob_words

    def encode(self, docs):
        relu = torch.nn.ReLU()
        # sigmoid = torch.nn.Sigmoid()
        hidden = relu(self.lnr_h2(relu(self.lnr_h1(docs))))
        mu = self.lnr_mu(hidden)
        sigma = self.lnr_sigma(hidden)
        return mu, sigma

    def decode(self, latent):
        # Listening to the advice on the torch.nn.Softmax; we use
        # LogSoftmax since we'll use NLLLose. Rather than
        # multiplication of each word prob and then taking log, we
        # compute log and then sum the probabilities.
        log_softmax = torch.nn.LogSoftmax(dim=1)
        log_prob_words = log_softmax(self.lnr_rec_doc(latent))
        log_nn_prob_words = log_softmax(self.lnr_nn_rec_doc(latent))
        return log_prob_words, log_nn_prob_words


def doc_rec_loss(log_prob_words, doc_batch):
    doc_mask = doc_batch.clone()
    doc_mask[torch.where(doc_mask != 0)] = 1.0
    rel_log_prob_words = torch.mul(log_prob_words, doc_mask)
    return -torch.mean(torch.sum(rel_log_prob_words, dim=1))


def doc_nn_rec_loss(log_nn_prob_words, knn_batch, train_docs):
    word_mask = None
    for knn in knn_batch:
        nn_docs = torch.from_numpy(train_docs[knn].todense())
        nn_mask = torch.sum(nn_docs != 0, dim=0) != 0
        nn_mask = nn_mask.reshape(1, len(nn_mask))
        if word_mask is None:
            word_mask = nn_mask
        else:
            word_mask = torch.cat((word_mask, nn_mask))
    rel_log_nn_prob_words = torch.mul(log_nn_prob_words, word_mask)
    return -torch.mean(torch.sum(rel_log_nn_prob_words, dim=1))


def binarize(means, threshold):
    hashes = means.clone()
    ones_i = hashes > threshold
    zeros_i = hashes <= threshold
    hashes[ones_i] = 1
    hashes[zeros_i] = 0
    return hashes


def encode_with_batches(docs, model, bsize):
    num_iter = int(np.ceil(docs.shape[0] / bsize))
    means = None
    for i in range(num_iter):
        batch = docs[i * bsize:(i+1) * bsize].todense()
        batch = torch.from_numpy(batch).double()
        mu, _ = model.encode(batch)
        if means is None:
            means = mu
        else:
            means = torch.cat((means, mu))
    return means


def hamming_score(test, train):
    return torch.sum(test == train, dim=1)


def test(train_docs, train_cats, test_docs, test_cats, model, bsize=100,
         k=100):
    model.eval()
    with torch.no_grad():
        train_means = encode_with_batches(train_docs, model, bsize)
        test_means = encode_with_batches(test_docs, model, bsize)
        threshold = torch.median(train_means, dim=0).values
        train_hash = binarize(train_means, threshold)
        test_hash = binarize(test_means, threshold)
        prec_sum = 0.0
        for i, th in enumerate(test_hash):
            hd = hamming_score(th.repeat(train_hash.shape[0], 1),
                               train_hash)
            _, topk_i = torch.topk(hd, k)
            rel = 0
            rel_cat = torch.from_numpy(test_cats[i].todense())
            for di in topk_i:
                train_cat = torch.from_numpy(train_cats[di].todense())
                rel += torch.sum(torch.mul(train_cat, rel_cat)).item()

            prec_sum += rel / k
        return prec_sum / test_hash.shape[0]


def train(train_docs, train_cats, train_knn, cv_docs, cv_cats, bitsize,
          epoch=30, bsize=100, lr=1e-3, latent_size=1000):
    nsize, lexsize = train_docs.shape
    num_iter = nsize // bsize
    model = NbrReg(lexsize, bitsize, h_size=latent_size)
    model.double()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    norm = tdist.Normal(0, 1)
    best_prec = 0.0
    for e in range(epoch):
        model.train()
        losses = []
        for i in range(num_iter):
            print(f"Epoch: {e + 1}/{epoch}, Iteration: {i + 1}/{num_iter}",
                  end="\r")
            batch_i = np.random.choice(nsize, bsize)
            np_batch = train_docs[batch_i].todense()
            doc_batch = torch.from_numpy(np_batch).double()
            knn_batch = train_knn[batch_i]
            optim.zero_grad()
            qdist, log_prob_words, log_nn_prob_words = model(doc_batch)
            doc_rl = doc_rec_loss(log_prob_words, doc_batch)
            doc_nn_rl = doc_nn_rec_loss(log_nn_prob_words, knn_batch,
                                        train_docs)
            kl_loss = tdist.kl_divergence(qdist, norm)
            kl_loss = torch.mean(torch.sum(kl_loss, dim=1))
            loss = doc_rl + doc_nn_rl + kl_loss
            losses.append(loss.item())
            loss.backward()
            optim.step()
        avg_loss = np.mean(losses)
        avg_prec = test(train_docs, train_cats, cv_docs, cv_cats, model)
        best_prec = max(avg_prec, best_prec)
        print(f"Epoch {e + 1}: Avg Loss: {avg_loss}, Avg Prec: {avg_prec}")
    return model, best_prec


def get_train_knn(fname, k=20):
    knn = None
    with open(fname) as fs:
        lines = fs.readlines()

    for l in lines:
        _, nns_str = l.rstrip().split(":", maxsplit=1)
        nns = [int(n) for n in nns_str.split(",")]
        nns = torch.tensor(nns[1:k + 1]).reshape(1, k)
        if knn is None:
            knn = nns
        else:
            knn = torch.cat((knn, nns))

    return knn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Input data")
    parser.add_argument("knn", help="Train knn data")
    parser.add_argument("--train", type=str, help="File to save trained model")
    parser.add_argument("--epoch", default=30, type=int)
    parser.add_argument("--test", type=str, help="Trained model")
    args = parser.parse_args()
    # data = np.load(args.data, allow_pickle=True)
    # docs = data["docs"].item()
    # categories = data["categories"]

    # nsize = docs.shape[0]
    # train_size = nsize // 10 * 8
    # cv_size = (nsize - train_size) // 2
    # test_size = nsize - train_size - cv_size

    # train_docs = docs[:train_size]
    # train_cats = categories[:train_size]
    # cv_start = train_size + cv_size
    # cv_docs = docs[train_size:cv_start]
    # cv_cats = categories[train_size:cv_start]
    # test_docs = docs[cv_start:cv_start + test_size]
    # test_cats = categories[cv_start:cv_start + test_size]

    data = scipy.io.loadmat(args.data)
    train_docs = data["train"]
    train_cats = data["gnd_train"]
    test_docs = data["test"]
    test_cats = data["gnd_test"]
    cv_docs = data["cv"]
    cv_cats = data["gnd_cv"]
    train_knn = get_train_knn(args.knn)
    if args.test:
        model = NbrReg(train_docs.shape[1], 32)
        model.double()
        model.load_state_dict(torch.load(args.test))
        avg_prec = test(train_docs, train_cats, test_docs, test_cats, model)
        print(f"Avg precision: {avg_prec}")
    else:
        model, best_prec = train(train_docs, train_cats, train_knn, cv_docs,
                                 cv_cats, 32, epoch=args.epoch)
        torch.save(model.state_dict(), args.train)
    return 0


if __name__ == "__main__":
    exit(main())
