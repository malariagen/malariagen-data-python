# Implementation of median joining network following Bandelt, Forster and Röhl (1999)
# https://pubmed.ncbi.nlm.nih.gov/10331250/
# https://doi.org/10.1093/oxfordjournals.molbev.a026036


import numba
import numpy as np


def _minimum_spanning_network(dist, max_dist=None):

    # keep only the upper triangle of the distance matrix, to avoid adding the same
    # edge twice
    dist = np.triu(dist)

    # setup the output array of links between nodes
    edges = np.zeros_like(dist)

    # setup an array of alternate links
    alternate_edges = np.zeros_like(dist)

    # intermediate variable - assignment of haplotypes to clusters (a.k.a. sub-networks)
    # initially each distinct haplotype is in its own cluster
    cluster = np.arange(dist.shape[0])

    # start with haplotypes separated by a single mutation
    step = 1

    # iterate until all haplotypes in a single cluster, or max_dist reached
    while len(set(cluster)) > 1 and (max_dist is None or step <= max_dist):

        # keep track of which clusters have been merged at this height
        merged = set()

        # remember what cluster assignments were at the previous height
        prv_cluster = cluster.copy()

        # iterate over all pairs where distance equals current step size
        for i, j in zip(*np.nonzero(dist == step)):

            # current cluster assignment for each haplotype
            a = cluster[i]
            b = cluster[j]

            # previous cluster assignment for each haplotype
            pa = prv_cluster[i]
            pb = prv_cluster[j]

            # check to see if both nodes already in the same cluster
            if a != b:

                # nodes are in different clusters, so we can merge (i.e., connect) the
                # clusters

                edges[i, j] = dist[i, j]
                edges[j, i] = dist[i, j]

                # merge clusters
                c = cluster.max() + 1
                loc_a = cluster == a
                loc_b = cluster == b
                cluster[loc_a] = c
                cluster[loc_b] = c
                merged.add(tuple(sorted([pa, pb])))

            elif tuple(sorted([pa, pb])) in merged or step == 1:

                # the two clusters have already been merged at this level, this is an
                # alternate connection
                # N.B., special case step = 1 because no previous cluster assignments
                # (TODO really?)

                alternate_edges[i, j] = dist[i, j]
                alternate_edges[j, i] = dist[i, j]

        # increment step
        step += 1

    return edges, alternate_edges


def _pairwise_haplotype_distance(h, metric="hamming"):
    import scipy.spatial

    assert metric in ["hamming", "jaccard"]
    dist = scipy.spatial.distance.pdist(h.T, metric=metric)
    dist *= h.shape[0]
    dist = scipy.spatial.distance.squareform(dist)
    # N.B., np.rint is **essential** here, otherwise can get weird rounding errors
    dist = np.rint(dist).astype("i8")
    return dist


def _mjn_remove_obsolete(h, orig_n_haplotypes, max_dist):
    n_removed = None
    edges = alt_edges = None

    while n_removed is None or n_removed > 0:

        # step 1 - compute distance
        dist = _pairwise_haplotype_distance(h, metric="hamming")

        # step 2 - construct the minimum spanning network
        edges, alt_edges = _minimum_spanning_network(dist, max_dist=max_dist)
        all_edges = edges + alt_edges

        # step 3 - remove obsolete sequence types
        loc_keep = np.ones(h.shape[1], dtype=bool)
        for i in range(orig_n_haplotypes, h.shape[1]):
            n_connections = np.count_nonzero(all_edges[i])
            if n_connections <= 2:
                loc_keep[i] = False
        n_removed = np.count_nonzero(~loc_keep)
        h = h[:, loc_keep]

    return h, edges, alt_edges


@numba.njit
def _uvw_consensus(h, max_allele):
    # here we form the consensus of three haplotypes, by taking the most common
    # allele at each site
    m = h.shape[0]
    n = h.shape[1]
    out = np.zeros(m, dtype=np.int8)
    ac = np.zeros(max_allele + 1, dtype=np.int32)
    for i in range(m):
        for j in range(n):
            allele = h[i, j]
            ac[allele] += 1
        consensus_allele = np.argmax(ac)
        out[i] = consensus_allele
        ac[:] = 0
    return out


def median_joining_network(h, max_dist=None, max_allele=1):

    # setup
    h = np.asarray(h)
    orig_n_haplotypes = h.shape[1]
    n_medians_added = None

    while n_medians_added is None or n_medians_added > 0:

        # steps 1-3

        h, edges, alt_edges = _mjn_remove_obsolete(
            h, orig_n_haplotypes=orig_n_haplotypes, max_dist=max_dist
        )
        all_edges = edges + alt_edges

        # step 4 - add median vectors

        # iterate over all triplets
        n = h.shape[1]
        seen = set([hash(h[:, i].tobytes()) for i in range(n)])
        new_haps = list()
        for i in range(n):
            for j in range(i + 1, n):
                if all_edges[i, j]:
                    for k in range(n):
                        if all_edges[i, k] or all_edges[j, k]:
                            uvw = h[:, [i, j, k]]
                            x = _uvw_consensus(uvw, max_allele)
                            x_hash = hash(x.tobytes())
                            # test if x already in haps
                            if x_hash not in seen:
                                new_haps.append(x)
                                seen.add(x_hash)
        n_medians_added = len(new_haps)
        if n_medians_added:
            new_haps = np.column_stack(new_haps)
            h = np.concatenate([h, new_haps], axis=1)

    # final pass
    h, edges, alt_edges = _mjn_remove_obsolete(
        h, orig_n_haplotypes=orig_n_haplotypes, max_dist=max_dist
    )
    return h, edges, alt_edges


def _mjn_graph_nodes(
    graph_nodes,
    ht_distinct,
    ht_distinct_mjn,
    ht_counts,
    ht_color_counts,
    color,
    color_values,
    edges,
    node_size_factor,
    anon_width,
):
    for i in range(ht_distinct_mjn.shape[1]):

        if i < ht_distinct.shape[1]:
            # original haplotype

            n = ht_counts[i]
            connected = np.any((edges[i] > 0) | (edges[:, i] > 0))
            if n == 1 and not connected:
                # don't show unconnected singletons
                continue

            # calculate width from number of items - make width proportional to area
            node_width = np.sqrt(n * node_size_factor)

            # create graph node
            graph_node = {
                "id": i,
                "count": n,
                "width": node_width,
            }

            # add color data
            if color:
                cc = ht_color_counts[i]
                for cv in color_values:
                    graph_node[cv] = cc.get(cv, 0) * 100 / n

        else:
            # not an original haplotype, inferred during network building
            graph_node = {
                "id": i,
                "count": 0,
                "width": anon_width,
            }

        graph_nodes.append(graph_node)


def _mjn_graph_edges(
    graph_edges,
    graph_nodes,
    edges,
    anon_width,
):
    for i in range(edges.shape[0]):

        for j in range(edges.shape[1]):

            # lookup distance between nodes i and j
            sep = edges[i, j]

            if sep == 1:
                # simple case, direct edge from node i to j
                graph_edge = {
                    "id": f"edge_{i}_{j}",
                    "source": i,
                    "target": j,
                }
                graph_edges.append(graph_edge)

            elif sep > 1:
                # tricky case, need to add some anonymous nodes to represent
                # intermediate steps

                # add first intermediate node
                graph_node = {
                    "id": f"anon_{i}_{j}_0",
                    "count": 0,
                    "width": anon_width,
                }
                graph_nodes.append(graph_node)

                # add edge from node i to first intermediate
                graph_edge = {
                    "id": f"edge_{i}_{j}_0",
                    "source": i,
                    "target": f"anon_{i}_{j}_0",
                }
                graph_edges.append(graph_edge)

                # add further intermediate nodes as necessary
                for k in range(1, sep - 1):
                    source = f"anon_{i}_{j}_{k-1}"
                    target = f"anon_{i}_{j}_{k}"
                    graph_node = {
                        "id": target,
                        "count": 0,
                        "width": anon_width,
                    }
                    graph_nodes.append(graph_node)
                    graph_edge = {
                        "id": f"edge_{i}_{j}_{k}",
                        "source": source,
                        "target": target,
                    }
                    graph_edges.append(graph_edge)

                # add edge from final intermediate node to node j
                source = f"anon_{i}_{j}_{sep-2}"
                target = j
                graph_node = {
                    "id": source,
                    "count": 0,
                    "width": anon_width,
                }
                graph_nodes.append(graph_node)
                graph_edge = {
                    "id": f"edge_{i}_{j}_{sep-1}",
                    "source": source,
                    "target": target,
                }
                graph_edges.append(graph_edge)


def mjn_graph(
    ht_distinct,
    ht_distinct_mjn,
    ht_counts,
    ht_color_counts,
    color,
    color_values,
    edges,
    alt_edges,
    node_size_factor,
    anon_width,
):
    graph_nodes = []
    graph_edges = []
    _mjn_graph_nodes(
        graph_nodes=graph_nodes,
        ht_distinct=ht_distinct,
        ht_distinct_mjn=ht_distinct_mjn,
        ht_counts=ht_counts,
        ht_color_counts=ht_color_counts,
        color=color,
        color_values=color_values,
        edges=edges,
        node_size_factor=node_size_factor,
        anon_width=anon_width,
    )
    _mjn_graph_edges(
        graph_edges=graph_edges,
        graph_nodes=graph_nodes,
        edges=edges,
        anon_width=anon_width,
    )
    _mjn_graph_edges(
        graph_edges=graph_edges,
        graph_nodes=graph_nodes,
        edges=alt_edges,
        anon_width=anon_width,
    )
    return graph_nodes, graph_edges
