# Rate Limiting LLM APIs

Hosted LLM APIs enforce limits on how many requests and tokens a client may send per minute, often abbreviated RPM and TPM. The limits protect shared GPU capacity, and exceeding them returns an HTTP 429 response. A production client must treat 429 as normal operating behavior rather than as an exceptional failure.

The standard remedy is retry with exponential backoff and jitter. After a failure the client waits a base delay, doubles it after each subsequent failure, and adds a random offset so that many clients do not retry in lockstep and create a thundering herd. Many providers return a Retry-After header, which should be honored when present. A cap on total attempts prevents a request from retrying forever.

Backoff only handles the symptom. To avoid hitting the limit at all, a client can meter its own traffic with a token bucket. The bucket fills at a steady rate equal to the allowed throughput and each request consumes tokens proportional to its size; when the bucket is empty, requests queue until it refills. Because LLM limits count tokens rather than requests, the bucket must estimate prompt and completion tokens before sending.

Concurrency control complements the bucket: a semaphore bounds the number of in-flight requests so that a burst of traffic does not open hundreds of connections at once. Batching several small prompts into one call and caching responses to identical prompts both reduce load further.

Finally, monitor your limit usage. Dashboards that track 429 rates, queue depth, and p99 latency reveal when it is time to request a higher quota or add a second provider as a fallback.
