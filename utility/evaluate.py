from .metrics import *
from .parser_TMHKG import parse_args

import torch
import numpy as np
import multiprocessing
import heapq
from time import time


# Global variables for worker processes
support_user_set = None
query_user_set = None
n_pois = None
Ks = None


def init_worker(_support_set, _query_set, _n_pois, _Ks):

    global support_user_set, query_user_set, n_pois, Ks
    support_user_set = _support_set
    query_user_set = _query_set
    n_pois = _n_pois
    Ks = _Ks


def test_one_user(x):

    # x[0]: rating_scores, x[1]: user_id
    rating = x[0]
    u = x[1]

    try:
        training_pois = support_user_set[u]
    except Exception:
        training_pois = []

    user_pos_query = query_user_set[u]

    all_pois = set(range(n_pois))
    test_pois = list(all_pois - set(training_pois))

    poi_score = {p: rating[p] for p in test_pois}
    K_max = max(Ks)
    K_max_poi_score = heapq.nlargest(K_max, poi_score, key=poi_score.get)

    r = [1 if p in user_pos_query else 0 for p in K_max_poi_score]

    precision, recall, ndcg = [], [], []
    for K in Ks:
        recall.append(recall_at_k(r, K, len(user_pos_query)))
        ndcg.append(ndcg_at_k(r, K))

    return {"recall": np.array(recall), "ndcg": np.array(ndcg)}


def test(model, support_tasks, query_tasks, model_data, all_history_tensor):

    _n_users = model_data["n_users"]
    _n_pois = model_data["n_items"]

    args = parse_args()
    _Ks = eval(args.Ks)
    device = (
        torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    )
    BATCH_SIZE = args.test_batch_size

    user_final_emb, poi_final_emb = model.generate(all_history_tensor)

    pool = multiprocessing.Pool(
        cores,
        initializer=init_worker,
        initargs=(support_tasks, query_tasks, _n_pois, _Ks),
    )

    test_users = list(query_tasks.keys())
    n_test_users = len(test_users)
    n_user_batches = n_test_users // BATCH_SIZE + 1

    result = {"recall": np.zeros(len(_Ks)), "ndcg": np.zeros(len(_Ks))}
    count = 0

    for u_batch_id in tqdm(range(n_user_batches), desc="Evaluating"):
        start = u_batch_id * BATCH_SIZE
        end = (u_batch_id + 1) * BATCH_SIZE
        user_list_batch = test_users[start:end]

        if not user_list_batch:
            continue

        user_batch = torch.LongTensor(user_list_batch).to(device)
        u_g_embeddings = user_final_emb[user_batch]

        rate_batch = model.rating(u_g_embeddings, poi_final_emb).detach().cpu()

        user_batch_rating_uid = zip(rate_batch.numpy(), user_list_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        count += len(batch_result)

        for re in batch_result:
            result["recall"] += re["recall"]
            result["ndcg"] += re["ndcg"]

    assert count == n_test_users
    pool.close()

    final_result = {
        "recall": (result["recall"] / n_test_users).tolist(),
        "ndcg": (result["ndcg"] / n_test_users).tolist(),
    }
    return final_result
