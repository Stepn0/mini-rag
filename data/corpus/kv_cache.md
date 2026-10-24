# The KV Cache

Autoregressive decoding generates one token at a time. At each step the model must attend over every earlier token, which means recomputing the keys and values for the whole prefix. Without optimization, generating a sequence of length n costs O(n^2) attention work per layer across the generation.

The KV cache removes this redundancy. Because the key and value projections of past tokens never change, they are computed once and stored. At each new decoding step only the query, key, and value for the newest token are computed; the new key and value are appended to the cache, and attention runs against the cached tensors. Each step then costs O(n) instead of O(n^2).

The cost is memory. For every layer and every attention head, the cache holds two vectors per token. For a model with 32 layers, 32 heads, head dimension 128 and 16-bit precision, a single token consumes 32 * 2 * 32 * 128 * 2 bytes, about 512 KB. A 32k-token context therefore needs roughly 16 GB just for the cache, which is why serving long contexts is memory bound rather than compute bound.

Several techniques shrink the cache. Grouped-query attention shares one key-value head among several query heads, dividing the cache size by the group factor. Multi-query attention takes this to the extreme with a single shared KV head. PagedAttention, popularized by vLLM, stores the cache in fixed-size blocks that can be allocated on demand, which eliminates fragmentation and allows prefixes shared between requests to be cached once. Quantizing the cache to 8 or 4 bits is another common trade.
