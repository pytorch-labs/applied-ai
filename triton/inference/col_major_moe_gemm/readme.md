Triton kernel supporting and accelerating MoE inference (Mixtral).
This kernel was contributed by IBM Research.

This kernel showcases moving the weights into col-major format to accelerate inference.
See blog post: (link pending)

v0 = grouped MM
v1 = SplitK MM
v2 = Col Major MM

This requires vLLM to be installed to run.
