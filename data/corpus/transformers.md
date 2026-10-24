# The Transformer Architecture

The Transformer is a neural network architecture built around attention instead of recurrence or convolution. Introduced for machine translation, it is now the backbone of nearly every large language model. A Transformer processes an entire sequence of tokens in parallel, which makes training far more efficient on GPUs than the step-by-step computation of recurrent networks.

The model is a stack of identical blocks. Each block has two sub-layers: a multi-head self-attention layer that lets every token gather information from every other token, and a position-wise feed-forward network that transforms each token independently. Residual connections wrap both sub-layers, and layer normalization keeps activations stable as the stack grows deeper.

Because attention is permutation-invariant, the model needs an explicit signal about token order. Positional encodings, either fixed sinusoids or learned vectors, are added to the input embeddings so the network can tell the first token from the tenth. Many modern models instead use rotary position embeddings, which rotate query and key vectors by an angle that depends on position.

Transformers come in three flavors. Encoder-only models such as BERT read the whole input bidirectionally and are good at classification. Decoder-only models such as GPT generate text left to right using a causal mask that hides future tokens. Encoder-decoder models keep both halves and are common in translation and summarization.

The cost of self-attention grows quadratically with sequence length, which is why long-context inference has become a major engineering focus.
