# Quantization for LLM Inference

Quantization reduces the numerical precision of a model's weights, and sometimes its activations, so that the model uses less memory and runs faster. A 7-billion-parameter model stored in 16-bit floats occupies about 14 GB; the same model in 4-bit integers fits in roughly 3.5 GB, small enough for a consumer GPU or a laptop.

The simplest scheme is uniform integer quantization. A group of floating-point weights is mapped to integers using a scale factor and, optionally, a zero point: q = round(w / scale) + zero_point. Dequantization reverses the mapping. Using a separate scale for each small group of weights, such as 32 or 128 values, limits the damage caused by outliers.

Post-training quantization (PTQ) converts an already trained model without any further gradient updates. Methods such as GPTQ and AWQ use a small calibration set to decide how to round each weight so that layer outputs change as little as possible. Quantization-aware training (QAT) instead simulates low precision during fine-tuning, letting the model adapt to the rounding error; it is more accurate but far more expensive.

Weight-only quantization helps most when inference is memory-bandwidth bound, which is the typical case for small-batch decoding: fewer bytes need to move from GPU memory for each generated token. Quantizing activations as well, for example to INT8, lets the hardware use faster integer matrix units, but outlier activations make this harder and often require special handling.

Formats such as GGUF package quantized weights together with their scales for use in CPU-friendly runtimes like llama.cpp.
