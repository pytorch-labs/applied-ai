
**MoE (Mixture of Experts) GEMM Kernels**


Triton kernel supporting and accelerating MoE inference (Mixtral).
This kernel was contributed by IBM Research.

This kernel showcases the following optimizations:

* Column-Major Launch Schedule (L2 Cache Optimization)
* SplitK Work Decomposition (Parallel Work Strategy Optimization)

See blog post: (link pending)

* v0 = grouped MM
* v1 = SplitK MM
* v2 = Col Major MM

This requires vLLM to be installed to run.
