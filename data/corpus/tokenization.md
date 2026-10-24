# Tokenization and Byte-Pair Encoding

Language models do not read characters or words; they read tokens, integer IDs drawn from a fixed vocabulary. Tokenization converts raw text into this sequence of IDs, and its design affects everything from context length to how well a model handles code and non-English languages.

Byte-pair encoding (BPE) is the dominant algorithm. Training starts with a vocabulary of individual bytes, then repeatedly finds the most frequent adjacent pair of symbols in the corpus and merges it into a new symbol. After tens of thousands of merges, common words become single tokens while rare words are spelled out from smaller pieces. Because the base vocabulary is bytes, any string can be encoded and nothing is ever out of vocabulary.

Encoding new text applies the learned merges in order. Modern tokenizers first split text with a regular expression that separates words, numbers, punctuation, and whitespace, so that merges never cross those boundaries. A leading space is usually attached to the following word, which is why " cat" and "cat" are different tokens.

Vocabulary size is a trade-off. A larger vocabulary compresses text into fewer tokens, which shortens sequences and lowers inference cost, but it enlarges the embedding matrix and gives rare tokens fewer training examples. Typical sizes range from 32,000 to over 200,000 entries.

Tokenization has quirks worth knowing. Numbers may be split into unpredictable chunks, which hurts arithmetic. Languages that are underrepresented in the training corpus need many more tokens per sentence, making them slower and more expensive. Counting tokens before sending a prompt is essential for staying within context windows and rate limits.
