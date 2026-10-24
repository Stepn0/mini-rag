# HNSW: Hierarchical Navigable Small World Graphs

HNSW is the most widely used algorithm for approximate nearest-neighbor search. It organizes vectors into a multi-layer proximity graph in which each node is connected to a bounded number of close neighbors. The result is a structure that can be searched in roughly logarithmic time while keeping recall very high.

The index is a hierarchy of layers. The bottom layer contains every vector. Each layer above contains a random subset of the layer below, chosen with an exponentially decaying probability, so the top layer has only a handful of nodes. Upper layers act like an express highway: long edges let the search cross the space quickly, while the dense bottom layer refines the result locally.

A search starts at an entry point in the top layer and greedily moves to whichever neighbor is closest to the query. When no neighbor improves the distance, the search drops down to the next layer and repeats from the current best node. At the bottom layer the algorithm performs a beam search that keeps a candidate list of size ef_search; larger values explore more nodes and improve recall at the cost of latency.

Construction inserts vectors one at a time. Each new vector is assigned a maximum layer, then connected to its M nearest neighbors at each layer it belongs to, using a heuristic that prefers diverse neighbors over purely closest ones. The parameter M controls memory use and graph connectivity, and ef_construction controls build quality.

HNSW's main weakness is memory: the whole graph and all vectors generally live in RAM, and deletions are awkward because removing nodes can disconnect the graph.
