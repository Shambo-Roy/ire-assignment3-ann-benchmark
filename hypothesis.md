# Pre-Experiment Hypotheses
## CS4.406 — Assignment 3: Benchmarking Approximate Nearest Neighbour Search

**Status:** Proposed, pending approval and Git commit.

## 1. Part 3 — Effect of Compression on SIFT

### Research question

How does the number of Product Quantization (PQ) sub-vectors affect the recall, query throughput, and memory usage of IVF-PQ? Can exact reranking of the top 100 retrieved candidates improve the recall–speed trade-off?

### Hypothesis 3.1: Effect of PQ granularity on recall and memory

I hypothesize that increasing the number of PQ sub-vectors will generally improve nearest-neighbour recall because each sub-vector represents a smaller portion of the original vector, potentially allowing the quantizer to represent the data more accurately. However, increasing the number of sub-vectors will also increase the storage required for PQ codes when the number of bits per sub-vector remains fixed.

This improvement may eventually diminish because increasing the representation capacity does not guarantee a proportional improvement in retrieval accuracy.

**How it will be tested:** I will vary the number of PQ sub-vectors while keeping other IVF-PQ parameters fixed wherever possible. I will measure recall@10, query throughput (QPS), and index size for each configuration.

### Hypothesis 3.2: Effect of exact reranking on recall

I hypothesize that reranking the top 100 IVF-PQ candidates using exact Euclidean distances will improve or preserve recall@10 compared with returning the candidates in their approximate PQ ranking.

PQ approximates distances, which can cause relevant neighbours to be ranked incorrectly. Exact reranking can correct these ordering errors among the candidates already retrieved. However, it cannot recover a true nearest neighbour that is absent from the candidate pool.

**How it will be tested:** I will compare recall@10 before and after exact reranking, using the same candidate pool and evaluation queries. I will also measure how many exact ground-truth neighbours occur in the top 100 candidates before reranking.

### Hypothesis 3.3: Effect of reranking on query throughput

I hypothesize that exact reranking will reduce query throughput because it introduces additional exact-distance computations and ranking operations. Nevertheless, the recall improvement may justify this overhead if reranking achieves a better recall–speed trade-off.

**How it will be tested:** I will measure QPS and recall@10 with and without reranking. I will compare the configurations at similar recall levels where possible and report the additional reranking cost separately if the implementation permits reliable measurement.

## 2. Part 4 — Effect of Dimensionality on Synthetic Data

### Research question

How does increasing vector dimensionality affect the recall and query throughput of HNSW and IVF-Flat when the dataset size, query count, distance metric, and search settings are held constant?

### Hypothesis 4.1: Effect of dimensionality on nearest-neighbour search

I hypothesize that increasing dimensionality will cause Euclidean distances between random normalized vectors to become more concentrated around a common value. This may make it more difficult for approximate nearest-neighbour indexes to distinguish the true nearest neighbours from other candidates under a fixed search budget.

Consequently, I expect recall@10 to decrease as dimensionality increases for at least some of the tested index configurations. However, this outcome is not guaranteed: the effect may depend on the index structure and search parameters.

**How it will be tested:** I will generate seeded synthetic datasets with dimensionalities of 32, 64, 128, 256, and 512. The 128-dimensional dataset will serve as the control. Each setting will contain approximately 100,000 indexed vectors and 1,000 queries. Exact search will provide the ground truth. I will measure recall@10 and examine distance statistics, including the mean and standard deviation of distances and the gap between the 10th and 11th exact neighbours.

### Hypothesis 4.2: Different effects on HNSW and IVF-Flat

I hypothesize that HNSW and IVF-Flat will respond differently to increasing dimensionality because they use different candidate-selection mechanisms. HNSW navigates a graph of nearby vectors, whereas IVF-Flat selects clusters and computes exact distances within the selected inverted lists.

Increasing dimensionality may affect graph navigation and cluster-based candidate selection differently. Therefore, the relative recall and throughput of the two indexes may change across dimensionalities.

**How it will be tested:** I will compare HNSW and IVF-Flat at every dimensionality using fixed, documented search settings. I will measure recall@10 and QPS and compare each index's change in recall relative to its 128-dimensional control result.

## 3. Experimental safeguards

1. I will record the random seed, dataset configuration, index parameters, library versions, and hardware information.
2. I will use the same evaluation methodology and tie-aware recall implementation throughout the experiments.
3. I will report results for all tested configurations, including results that do not support the hypotheses.
4. I will distinguish measured results from explanations inferred after the experiments.
5. I will not rewrite these hypotheses after observing the results. Any necessary protocol changes will be documented separately with their justification.
6. I will commit this file before running either Part 3 or Part 4.

