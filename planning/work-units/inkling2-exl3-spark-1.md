# Portable two-Spark text runtime

Owning issue: https://github.com/kortexa-ai/inkling2-exl3-spark/issues/1

Use pinned public runtime sources, a reproducible SM121 Docker build, verified
rank-specific model downloads, and plain shell entry points for two DGX Sparks.
Keep managed service integration in models.server. Refuse busy GPUs and keep
launch cleanup owned by the parent process. Validate stock EXL3 codec fixtures,
Gloo residual transport and actual text generation on the pair. Target 512K
with tiled FP8 attention and preallocated global KV; report the full-model
validation limit separately. One million tokens is outside this work unit.
