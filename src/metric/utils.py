import numpy as np
import logging
import multiprocessing as mp
import faiss


class FaissIndex(object):
    def __init__(self, corpus_vecs, dim, l2_normalize=False, index_name="IndexFlatIP"):
        self.l2_normalize = l2_normalize
        if self.l2_normalize:
            faiss.normalize_L2(corpus_vecs)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(corpus_vecs.astype("float32"))

    def search(self, query_vecs, topk=50):
        if self.l2_normalize:
            faiss.normalize_L2(query_vecs)
        topk_scores, topk_indices = self.index.search(query_vecs.astype("float32"), topk)
        return topk_scores, topk_indices


def evaluate_metrics(user_embs,
                     item_embs,
                     train_user2items,
                     valid_user2items,
                     query_indices,
                     metrics,
                     num_workers=1):
    logging.info("Evaluating metrics for {} users.".format(len(user_embs)))
    metric_funcs = []
    max_topk = 0
    for metric in metrics:
        try:
            metric_funcs.append(eval(metric))
            max_topk = max(max_topk, int(metric.split("k=")[-1].strip(")")))
        except:
            raise NotImplementedError('metrics={} not implemented.'.format(metric))

    faiss_index = FaissIndex(item_embs, dim=item_embs.shape[-1])
    chunk_size = min(1000, int(np.ceil(len(user_embs) / float(num_workers))))
    pool = mp.Pool(processes=num_workers)
    results = []
    for idx in range(0, len(user_embs), chunk_size):
        chunk_user_embs = user_embs[idx: (idx + chunk_size), :]
        chunk_query_indices = query_indices[idx: (idx + chunk_size)]
        if num_workers > 1:
            results.append(pool.apply_async(evaluate_block,
                                            args=(chunk_user_embs, faiss_index, chunk_query_indices,
                                                  train_user2items, valid_user2items, metric_funcs, max_topk)))
        else:
            results += evaluate_block(chunk_user_embs, faiss_index, chunk_query_indices, train_user2items,
                                      valid_user2items, metric_funcs, max_topk)
    if num_workers > 1:
        pool.close()
        pool.join()
        results = [res.get() for res in results]
    average_result = np.average(np.array(results), axis=0).tolist()
    return_dict = dict(zip(metrics, average_result))
    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in zip(metrics, average_result)))
    return return_dict


def evaluate_block(user_embs, faiss_index, query_indices, train_user2items,
                   valid_user2items, metric_funcs, max_topk):
    # set to topk=500 here since the retrieval results may contain clicked items
    topk = 500
    if len(user_embs.shape) == 3:
        bsz, num_groups, hidden_dim = user_embs.shape
        topk = topk * num_groups
        user_embs_flat = user_embs.reshape(-1, user_embs.shape[-1])
        scores, indices = faiss_index.search(user_embs_flat, topk=topk)
        scores = scores.reshape(bsz, num_groups, topk)
        indices = indices.reshape(bsz, num_groups, topk)
        max_indices = np.argmax(scores, axis=1)
        batch_indices = np.arange(bsz)[:, None]
        width_indices = np.arange(topk)[None, :]
        scores = scores[batch_indices, max_indices, width_indices]
        indices = indices[batch_indices, max_indices, width_indices]
    else:
        scores, indices = faiss_index.search(user_embs, topk=500)
    mask = np.zeros((user_embs.shape[0], faiss_index.index.ntotal))
    for i, query_index in enumerate(query_indices):
        train_items = train_user2items[query_index]
        mask[i, train_items] = 1
    mask = np.take_along_axis(mask, indices, axis=1)  # ie, mask[np.arange(len(mask))[:, None], indices]
    scores += -1e9 * mask
    sorted_idxs = np.argsort(-scores, axis=1)
    topk_items = np.take_along_axis(indices, sorted_idxs, axis=1)[:, 0:max_topk]  # get max_topk for metrics
    true_items = [valid_user2items[query_index] for query_index in query_indices]
    chunk_results = [[func(preds, labels) for func in metric_funcs]
                     for preds, labels in zip(topk_items, true_items)]
    # if len(user_embs.shape) == 3:
    #     # search in multi-interest group
    #     bsz, num_groups, hidden_dim = user_embs.shape
    #     user_embs_flat = user_embs.reshape(bsz * num_groups, hidden_dim)
    #     scores, indices = faiss_index.search(user_embs_flat, topk=500 * num_groups)
    #     scores = scores.reshape(bsz, num_groups, 500 * num_groups)
    #     indices = indices.reshape(bsz, num_groups, 500 * num_groups)
    #
    #     # combine scores
    #     max_indices = np.argmax(scores, axis=1)
    #     batch_indices = np.arange(bsz)[:, None]
    #     width_indices = np.arange(500 * num_groups)[None, :]
    #     scores = scores[batch_indices, max_indices, width_indices]
    #     indices = indices[batch_indices, max_indices, width_indices]
    #
    #     mask = np.zeros((user_embs.shape[0], faiss_index.index.ntotal))
    #     for i, query_index in enumerate(query_indices):
    #         train_items = train_user2items[query_index]
    #         mask[i, train_items] = 1
    #     mask = np.take_along_axis(mask, indices, axis=1)  # ie, mask[np.arange(len(mask))[:, None], indices]
    #     scores += -1e9 * mask
    #     sorted_idxs = np.argsort(-scores, axis=1)
    #     topk_items = np.take_along_axis(indices, sorted_idxs, axis=1)[:, 0:max_topk]  # get max_topk for metrics
    #     true_items = [valid_user2items[query_index] for query_index in query_indices]
    #     chunk_results = [[func(preds, labels) for func in metric_funcs]
    #                      for preds, labels in zip(topk_items, true_items)]
    #
    #     # final_topk_items_all = []
    #     # for i, query_index in enumerate(query_indices):
    #     #     user_scores = scores[i].reshape(-1)  # shape: [num_groups * retrieval_topk]
    #     #     user_indices = indices[i].reshape(-1)  # shape: [num_groups * retrieval_topk]
    #     #     train_items_set = set(train_user2items[query_index])
    #     #
    #     #     item2best_score = {}
    #     #
    #     #     for item, sc in zip(user_indices, user_scores):
    #     #         if item not in train_items_set:
    #     #             if item not in item2best_score:
    #     #                 item2best_score[item] = sc
    #     #             else:
    #     #                 if sc > item2best_score[item]:
    #     #                     item2best_score[item] = sc
    #     #
    #     #     sorted_item_score_pairs = sorted(item2best_score.items(), key=lambda x: x[1], reverse=True)
    #     #     top_items = [x[0] for x in sorted_item_score_pairs[:max_topk]]
    #     #     final_topk_items_all.append(top_items)
    #     # chunk_results = []
    #     # for i, query_index in enumerate(query_indices):
    #     #     preds = final_topk_items_all[i]
    #     #     labels = valid_user2items[query_index]
    #     #     metrics_for_user = [func(preds, labels) for func in metric_funcs]
    #     #     chunk_results.append(metrics_for_user)
    # else:
    #     scores, indices = faiss_index.search(user_embs, topk=500)
    #     # mask out items already clicked in train data
    #     mask = np.zeros((user_embs.shape[0], faiss_index.index.ntotal))
    #     for i, query_index in enumerate(query_indices):
    #         train_items = train_user2items[query_index]
    #         mask[i, train_items] = 1
    #     mask = np.take_along_axis(mask, indices, axis=1)  # ie, mask[np.arange(len(mask))[:, None], indices]
    #     scores += -1e9 * mask
    #     sorted_idxs = np.argsort(-scores, axis=1)
    #     topk_items = np.take_along_axis(indices, sorted_idxs, axis=1)[:, 0:max_topk]  # get max_topk for metrics
    #     true_items = [valid_user2items[query_index] for query_index in query_indices]
    #     chunk_results = [[func(preds, labels) for func in metric_funcs]
    #                      for preds, labels in zip(topk_items, true_items)]
    return chunk_results


class Recall(object):
    """Recall metric."""

    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        recall = len(hit_items) / (len(true_items) + 1e-12)
        return recall


class nRecall(object):
    """Recall metric normalized with max 1 at topk, like nDCG"""

    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        recall = len(hit_items) / min(self.topk, len(true_items) + 1e-12)
        return recall


class Precision(object):
    """Precision metric."""

    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        precision = len(hit_items) / (self.topk + 1e-12)
        return precision


class F1(object):
    def __init__(self, k=1):
        self.precision_k = Precision(k)
        self.recall_k = Recall(k)

    def __call__(self, topk_items, true_items):
        p = self.precision_k(topk_items, true_items)
        r = self.recall_k(topk_items, true_items)
        f1 = 2 * p * r / (p + r + 1e-12)
        return f1


class DCG(object):
    """ Calculate discounted cumulative gain
    """

    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        true_items = set(true_items)
        dcg = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                dcg += 1 / np.log(2 + i)
        return dcg


class NDCG(object):
    """Normalized discounted cumulative gain metric."""

    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        dcg_fn = DCG(k=self.topk)
        idcg = dcg_fn(true_items[:self.topk], true_items)
        dcg = dcg_fn(topk_items, true_items)
        return dcg / (idcg + 1e-12)


class MRR(object):
    """MRR metric"""

    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        true_items = set(true_items)
        mrr = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                mrr += 1 / (i + 1.0)
        return mrr


class HitRate(object):
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        hit_rate = 1 if len(hit_items) > 0 else 0
        return hit_rate


class MAP(object):
    """
    Calculate mean average precision.
    """

    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        true_items = set(true_items)
        pos = 0
        precision = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                pos += 1
                precision += pos / (i + 1.0)
        return precision / (pos + 1e-12)
