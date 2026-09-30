import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def lead_n(sentences, n=3):
    if not sentences:
        return ""
    return " ".join(sentences[:n])


def textrank(sentences, n=3, damping=0.85, iters=40):
    if not sentences:
        return ""
    if len(sentences) <= n:
        return " ".join(sentences)

    vec = TfidfVectorizer()
    try:
        x = vec.fit_transform(sentences)
    except ValueError as error:
        if "empty vocabulary" not in str(error):
            raise
        return lead_n(sentences, n)
    sim = (x * x.T).toarray()
    np.fill_diagonal(sim, 0.0)
    row_sum = sim.sum(axis=1, keepdims=True)
    dangling = row_sum[:, 0] == 0
    row_sum[dangling] = 1.0
    trans = sim / row_sum
    trans[dangling] = 1.0 / len(sentences)

    scores = np.ones(len(sentences)) / len(sentences)
    teleport = (1.0 - damping) / len(sentences)
    for _ in range(iters):
        scores = damping * trans.T.dot(scores) + teleport

    top = sorted(np.argsort(-np.round(scores, 14), kind="stable")[:n])
    return " ".join(sentences[i] for i in top)
