# Evaluating RAG Systems

A retrieval-augmented generation system has two stages that can fail independently, so good evaluation measures them separately. If the retriever never surfaces the right passage, no generator can produce a grounded answer; if the retriever succeeds but the generator ignores the context, the pipeline still fails.

Retrieval is evaluated with classic ranking metrics against a set of questions labeled with the passages or documents that answer them. Recall@k asks whether a relevant document appears anywhere in the top k results. Mean reciprocal rank (MRR) averages 1/rank of the first relevant result, so it rewards putting the right document at the very top. Normalized discounted cumulative gain (nDCG) handles graded relevance when some passages are better than others.

Generation is evaluated on faithfulness and answer quality. Faithfulness, sometimes called groundedness, checks whether every claim in the answer is supported by the retrieved context; an unfaithful answer is a hallucination even if it happens to be true. Answer relevance checks whether the response actually addresses the question. Citation accuracy checks that each cited passage really supports the sentence that cites it.

Because reading hundreds of answers by hand is slow, teams often use an LLM as a judge, prompting a strong model to grade faithfulness or correctness. Judges are convenient but have known biases, such as preferring longer answers, so they should be calibrated against a human-labeled sample.

Finally, measure latency and cost per query. A pipeline that improves recall by one point while doubling latency may be a regression for users.
