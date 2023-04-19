import numpy as np
from pathlib import Path

from sklearn.cluster import KMeans
from collections import defaultdict

from .utils import preemptive_save


def cluster(embeds, indices, save_dir):
    save_dir = Path(save_dir)
    n = len(embeds)
    print(f"Clustering {n} embeddings")
    embeds = np.stack(embeds, 0)

    episode_dict = defaultdict(list)

    for i, idx in enumerate(indices):
        ep_id, step = idx
        episode_dict[ep_id].append((step, i))
    
    ep_lens = []
    for ep_id in episode_dict.keys():
        episode_dict[ep_id] = sorted(episode_dict[ep_id])
        ep_lens.append(len(episode_dict[ep_id]))
    
    ep_lens = sorted(ep_lens, reverse=True)
    k = ep_lens[5]
    kmeans = None
    labels = None

    LIMIT = 30

    while True:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(embeds)
        labels = kmeans.labels_

        num_conflicts = 0
        for ep in episode_dict.values():
            l = len(ep)
            for i in range(l - 1):
                for j in range(i + 1, l):
                    if labels[ep[i][1]] == labels[ep[j][1]]:
                        num_conflicts += 1

        print(f"k={k}, conflicts={num_conflicts}")
        if num_conflicts < n * 0.001 or k >= LIMIT:
            break
        k += 1

    print(f"Optimal k={k}")

    correct_distances = [[] for _ in range(k)]
    wrong_distances = [[] for _ in range(k)]
    for i in range(n):
        for j in range(k):
            dist = np.square(embeds[i] - kmeans.cluster_centers_[j]).sum()
            if j == labels[i]:
                correct_distances[j].append(dist)
            else:
                wrong_distances[j].append(dist)
    
    min_dist = -1
    max_dist = 1e9
    for j in range(k):
        cd = sorted(correct_distances[j], reverse=True)
        wd = sorted(wrong_distances[j])
        min_dist = max(min_dist, cd[int(len(cd) * 0.01)])
        max_dist = min(max_dist, wd[int(len(wd) * 0.01)])
        # print(j, min_dist, max_dist)
    
    sep_dist = (min_dist + max_dist) / 2
    print(min_dist, max_dist, (min_dist + max_dist) / 2)

    preemptive_save({
        "names": [f"achievement-{i}" for i in range(k)], 
        "centroids": kmeans.cluster_centers_,
        "threshold": sep_dist
    }, save_dir / "cluster.data", type="joblib")

    # preliminary graph
    orders = np.zeros((k, k), dtype=int)
    happens = np.zeros((k,), dtype=int)
    for ep in episode_dict.values():
        l = len(ep)
        for i in range(l):
            li = labels[ep[i][1]]
            happens[li] += 1
            for j in range(i + 1, l):
                lj = labels[ep[j][1]]
                orders[li][lj] += 1

    graph = np.zeros((k, k), dtype=int)

    for i in range(k):
        for j in range(k):
            connected = orders[i][j] / happens[j] > 0.97 and orders[j][i] / orders[i][j] < 0.001

            graph[i][j] = 1 if connected else 0
            # if connected:
            #     print(i, '->', j, end=', ')

    preemptive_save(graph, save_dir / "graph.data", type="joblib")

    # clean up edges
    def find_earliest(u):
        nl = [[u], []]
        all_nodes = set([u])
        d = 0
        cnt = 0
        while len(nl[d]) > 0:
            nl[1 - d] = []
            for v in nl[d]:
                for x in range(k):
                    if graph[x][v] and not x in all_nodes:
                        nl[1 - d].append(x)
                        all_nodes.add(x)
                        cnt += 1
            d = 1 - d
        return cnt
    
    key_edges = []
    for i in range(k):
        for j in range(k):
            if graph[i][j] == 1:
                earliest = find_earliest(j)
                graph[i][j] = 0
                connected = find_earliest(j) < earliest
                graph[i][j] = 1
                if connected:
                    key_edges.append((i, j))
                # if connected:
                #     print(i, '->', j)
            else:
                connected = False
            print(f"{1 if connected else 0}", end='\n' if j == k - 1 else ',')
    print(key_edges)

    return k