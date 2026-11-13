# Inkling 2 Small EXL3 on two DGX Sparks

Run the 194.05 GiB Inkling 2 Small EXL3 export across two 128 GB NVIDIA DGX
Sparks. Each host holds 24 layers. The head sends BF16 residual streams over
ordinary IP to the second host using Gloo. No RDMA configuration is required.

This is an experimental **text-only, single-request command-line runner**.
It loads both stages, answers one prompt and exits. Vision, speculative decoding,
concurrent requests and an HTTP API are outside this recipe.

## Requirements

- Two ARM64 DGX Sparks with working NVIDIA drivers, Docker and NVIDIA Container Toolkit.
- Bash, Python 3.12+, Git and passwordless noninteractive SSH from the head to the peer.
- At least 125 GiB free disk per host for its model shard, plus about 45 GiB for Docker/build space.
- Both GPUs idle. Stop any resident model before setup or inference.
- Internet access to Docker Hub, GitHub, PyPI and Hugging Face during setup/download.

The launch scripts refuse busy GPUs and used rendezvous ports. They never stop
other models. The parent process stops both containers on completion, failure,
a signal, timeout or the 6.5 GiB available-host-memory guard.

## Run

On the Spark that will be the head, replace `second-spark` with the SSH name or
`user@address` of the other Spark:

```bash
git clone https://github.com/kortexa-ai/inkling2-exl3-spark.git
cd inkling2-exl3-spark
./setup.sh second-spark
./download.sh second-spark
./start.sh second-spark --prompt 'What is 7 * 6? Reply with only the number.'
```

Setup clones this repository on the peer at `~/src/inkling2-exl3-spark`, builds
one image from pinned public sources and transfers that image to the peer.
Both image IDs must match. It uses two compiler jobs and needs 24 GiB available
host RAM on each Spark. It does not change drivers, services or Docker restart policies.
Both repository clones must be clean at the same commit.

Download stores only each host's selected files under `~/models/inkling2-exl3`.
Every weight is checked against the published SHA-256 inventory. Rerunning
download skips verified files and replaces incomplete or corrupt downloads.
The full export is on [Hugging Face](https://huggingface.co/kortexa-ai/Inkling-2-Small-EXL3-2.59bpw).

For a file prompt:

```bash
./start.sh second-spark --prompt-file ./prompt.txt --max-tokens 256
```

The launcher picks the head address and Gloo interfaces from the route to the
peer. For a particular network, pass `--head-address HEAD_IPV4`. The peer must
reach that address. The default rendezvous port is 18095; Gloo also opens
peer sockets, so allow traffic between the two hosts. This recipe is for a
trusted two-host network. It exposes no model HTTP endpoint.

`--timeout SECONDS` bounds the complete launch, including loading and generation
(default 86400). Logs are saved under `.runtime/logs/`. `--protocol-canary`
checks exact BF16 pipeline round trips without loading the weights.

## Context and memory

The serving target and default admission limit are **524,288 tokens (512K)**,
including prompt and generation. Scaled FP8 E4M3 KV uses per-token/head scales.
Four global caches plus the local histories cost about 4.20 GiB per host at
512K. Global storage is preallocated for the actual request length; tiled
attention avoids full K/V expansion across query heads and full score matrices.
The allocator uses expandable segments. Prefill chunks stay at or below 128 rows
to use the codec's direct quantized path.

**The full-model validation is 8K.** A separate 512K attention/cache capacity
check supports the memory plan; it does not establish full-model quality or
throughput at 512K. Long prompts can take hours with this sequential Python
MoE runner. The publisher's 1,048,576-token model maximum is outside this recipe;
1M is deferred until weight optimization and separate qualification.

The original two-Spark 8K trial used 8,128 prompt tokens and recovered early,
middle and late needles: `orchid, 47, nebula, 42`. It generated 13 tokens in
334.03 seconds including 324.92 seconds of prefill. Peak allocated memory was
92.23 / 98.20 GiB. These are baseline measurements, not a speed promise for
other software/hardware configurations. Held-out quantization metrics and
artifact provenance are in the Hugging Face model card.

## Reproducibility

The Dockerfile fixes an ARM64 public CUDA 13 / PyTorch 2.13 base by digest.
It builds unmodified MIT ExLlamaV3 at commit
`151539c77abc7ab7425d30da7a4e8e3c5c154e7b` for SM121 and fetches selected
Apache-2.0 SGLang kernels at commit
`237ba6baf2f2409f9d82d02fae1b909ea51ffb23`.
`source-pins.json` records hashes for every fetched build input.
No upstream inference server is started inside the image.

CPU tests inside the built image:

```bash
image=$(python3 -c 'import json; print(json.load(open(".runtime/setup.json"))["image"])')
docker run --rm --entrypoint python3 "$image" -m unittest discover -s /opt/inkling/tests -q
```

To run the GPU attention conformance and four-cache 512K capacity checks on an
idle Spark, add `--gpus all -e INKLING_GPU_TEST=1`. Optional independent codec
fixtures can be mounted and selected with `INKLING_FIXTURES`; these are not
required for inference.

Source code is Apache-2.0. Original model weights are Apache-2.0; EXL3 is MIT.
See LICENSE and NOTICE for attribution.
