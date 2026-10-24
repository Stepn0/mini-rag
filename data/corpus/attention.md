# Scaled Dot-Product Attention

Attention is a mechanism for computing a weighted average of values, where the weights are determined by how well a query matches a set of keys. In a Transformer, every token produces three vectors through learned linear projections: a query Q, a key K, and a value V.

The scores are computed as the dot product between a query and every key. These scores are divided by the square root of the key dimension d_k, which keeps the dot products from growing large as the dimension increases and prevents the softmax from saturating into near one-hot distributions. A softmax turns the scaled scores into a probability distribution, and the output is the sum of the values weighted by those probabilities. In matrix form: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

Multi-head attention runs this computation several times in parallel with different projections. Each head can specialize: one might track syntactic agreement while another follows coreference. The head outputs are concatenated and projected back to the model dimension.

In decoder models, a causal mask sets the scores for future positions to negative infinity before the softmax, so a token can only attend to itself and earlier tokens. Without this mask, the model would see the answer during training.

Attention is expensive: computing the full score matrix costs O(n^2) time and memory in the sequence length n. FlashAttention reorganizes the computation into tiles that fit in on-chip SRAM, avoiding materializing the full matrix and dramatically speeding up both training and inference.
