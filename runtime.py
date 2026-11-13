#!/usr/bin/env python3
"""Two-stage single-request CLI for a pair of DGX Sparks."""
import argparse,hashlib,json,os,time
from pathlib import Path
from datetime import timedelta
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch
import torch.distributed as dist
import model as full
import kernels

def memory_snapshot():
    free, total = torch.cuda.mem_get_info()
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    return {"allocated_gib": torch.cuda.memory_allocated()/kernels.GIB,
            "peak_allocated_gib": torch.cuda.max_memory_allocated()/kernels.GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved()/kernels.GIB,
            "device_free_gib": free/kernels.GIB, "device_total_gib": total/kernels.GIB,
            "host_available_gib": available/kernels.GIB}

def discard_file_cache(path):
    # These immutable checkpoint pages are no longer needed after a CUDA copy.
    # Avoid a host-wide cache flush or a privileged launcher on unified memory.
    with path.open("rb") as source:
        os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

def verify_rank_file(path, receipt):
    if path.stat().st_size != receipt["bytes"]:
        raise ValueError(f"Checkpoint size differs: {path}")
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4*1024**2):
            value.update(block)
        os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    if value.hexdigest() != receipt["sha256"]:
        raise ValueError(f"Checkpoint hash differs: {path}")

def check_load_budget(receipt, max_context, chunk):
    snapshot = memory_snapshot()
    # NVIDIA's GB10 UMA guidance says cudaMemGetInfo excludes reclaimable
    # filesystem cache. MemAvailable is the non-swapping allocation budget.
    unified = "GB10" in torch.cuda.get_device_name()
    available = snapshot["host_available_gib" if unified else "device_free_gib"]*kernels.GIB
    cache = (4*max_context+20*1023)*8*(128+4)*2
    required = receipt["resident_weight_bytes"]+cache+12.5*kernels.GIB
    result = {"accounting": "GB10 MemAvailable" if unified else "CUDA free",
              "available_gib": available/kernels.GIB, "required_gib": required/kernels.GIB,
              "memory": snapshot}
    if available < required:
        raise RuntimeError("Trial load budget does not fit: "+json.dumps(result))
    return result

def send_json(value, destination):
    payload = json.dumps(value, allow_nan=False).encode()
    if len(payload) > 65536:
        raise ValueError("Transport control message too large")
    dist.send(torch.tensor([len(payload)], dtype=torch.long), destination)
    dist.send(torch.tensor(list(payload), dtype=torch.uint8), destination)

def receive_json(source):
    length = torch.zeros(1, dtype=torch.long)
    dist.recv(length, source)
    if not 0 < int(length) <= 65536:
        raise ValueError("Invalid transport control length")
    payload = torch.empty(int(length), dtype=torch.uint8)
    dist.recv(payload, source)
    return json.loads(bytes(payload.tolist()))

def send_state(state, destination):
    for tensor in (state.partial, *state.blocks):
        dist.send(tensor.contiguous().cpu(), destination)

def receive_state(tokens, blocks, source, device):
    if not 0 < tokens <= 512 or not 0 <= blocks <= 8:
        raise ValueError("Invalid residual transport shape")
    values = []
    for _ in range(blocks + 1):
        value = torch.empty((tokens, 4608), dtype=torch.bfloat16)
        dist.recv(value, source)
        values.append(value.to(device))
    return full.Residual(values[0], values[1:])

def protocol_canary(rank):
    for tokens, blocks in ((7, 4), (128, 4), (1, 8)):
        if rank == 0:
            torch.manual_seed(115 + tokens)
            state = full.Residual(torch.randn(tokens, 4608).bfloat16(),
                                  [torch.randn(tokens, 4608).bfloat16() for _ in range(blocks)])
            send_json({"tokens": tokens, "blocks": blocks}, 1)
            send_state(state, 1)
            echoed = receive_state(tokens, blocks, 1, "cpu")
            assert torch.equal(echoed.partial, state.partial)
            assert all(torch.equal(a, b) for a, b in zip(echoed.blocks, state.blocks))
        else:
            header = receive_json(0)
            state = receive_state(header["tokens"], header["blocks"], 0, "cpu")
            send_state(state, 0)
    return {"rank": rank, "transport": "Gloo CPU BF16", "residual_roundtrip_exact": True}

class ResidentReader(full.Checkpoint):
    def __init__(self, path):
        super().__init__(path)
        self.gpu_globals = {}

    def tensor(self, key, device="cpu"):
        if torch.device(device).type == "cuda" and not key.startswith("model.llm.layers."):
            if key not in self.gpu_globals:
                self.gpu_globals[key] = super().tensor(key, device)
                discard_file_cache(self.root/self.index[key])
            return self.gpu_globals[key]
        return super().tensor(key, device)


class Stage:
    def __init__(self, checkpoint, rank, max_context=524288):
        self.rank = rank
        self.max_context = max_context
        self.reader = ResidentReader(checkpoint)
        self.ops = full.Ops()
        self.layers = []
        self.first, self.last = (0, 24) if rank == 0 else (24, 48)
        for index in range(self.first, self.last):
            layer = full.Layer(self.reader, index, self.ops)
            layer.load_packed()
            self.layers.append(layer)
            for path in [f"dense/layer-{index:02d}.safetensors",
                         *[b["file"] for b in self.reader.receipt["layers"][str(index)]["batches"]]]:
                discard_file_cache(self.reader.root/path)
            torch.cuda.empty_cache()
            print(json.dumps({"rank": rank, "loaded_layer": index,
                              "allocated_gib": torch.cuda.memory_allocated()/kernels.GIB,
                              "memory": memory_snapshot()}), flush=True)
        self.clear()

    def clear(self, capacity=None):
        capacity = self.max_context if capacity is None else capacity
        if not 1 <= capacity <= self.max_context:
            raise ValueError("Invalid request KV capacity")
        self.caches = [{"capacity": capacity} for _ in self.layers]

    def forward(self, state):
        for layer, cache in zip(self.layers, self.caches):
            layer.forward(state, cache, fp8_cache=True)
        # Increasing global-attention shapes otherwise retain each previous
        # scratch allocation in the CUDA pool, consuming shared host headroom.
        torch.cuda.empty_cache()
        return state

def sample(logits, temperature, top_p):
    if temperature <= 0:
        return int(logits.argmax())
    values, indices = (logits / temperature).sort(descending=True)
    probabilities = values.softmax(-1)
    remove = probabilities.cumsum(-1) - probabilities > top_p
    probabilities.masked_fill_(remove, 0)
    probabilities /= probabilities.sum()
    return int(indices[torch.multinomial(probabilities, 1)])

def worker(stage):
    while True:
        command = receive_json(0)
        if command["command"] == "stop":
            return
        if command["command"] == "reset":
            stage.clear(command["capacity"])
            torch.manual_seed(command["seed"])
            send_json({"reset": True}, 0)
            continue
        if command["command"] != "forward":
            raise ValueError("Unknown pipeline command")
        state = receive_state(command["tokens"], command["blocks"], 0, "cuda")
        stage.forward(state)
        if command["logits"]:
            values = full.logits(stage.reader, state, positions=torch.tensor([-1], device="cuda"))[0]
            token = sample(values, command["temperature"], command["top_p"])
            send_json({"token": token, "memory": memory_snapshot()}, 0)
        else:
            send_json({"prefill": True}, 0)

def generate(stage, args):
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.checkpoint / "auxiliary/tokenizer.json"))
    tokenizer.chat_template = (args.checkpoint / "auxiliary/chat_template.jinja").read_text()
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
        tokenize=False, reasoning_effort="none", add_generation_prompt=True)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if len(ids) + args.max_tokens > args.max_context:
        raise ValueError("Prompt plus generation exceeds this bounded trial's context admission limit")
    capacity = len(ids)+args.max_tokens
    stage.clear(capacity)
    send_json({"command": "reset", "seed": args.seed, "capacity": capacity}, 1)
    assert receive_json(1)["reset"]

    def forward(tokens, want_logits):
        state = full.Residual(full.embed(stage.reader, tokens))
        stage.forward(state)
        send_json({"command": "forward", "tokens": len(tokens), "blocks": len(state.blocks),
                   "logits": want_logits, "temperature": args.temperature, "top_p": args.top_p}, 1)
        send_state(state, 1)
        return receive_json(1)

    started = time.monotonic()
    last = None
    for offset in range(0, len(ids), args.chunk):
        last = forward(ids[offset:offset+args.chunk], offset+args.chunk >= len(ids))
        print(json.dumps({"rank": 0, "prefill_tokens": min(offset+args.chunk, len(ids)),
                          "prompt_tokens": len(ids), "elapsed_seconds": time.monotonic()-started,
                          "memory": memory_snapshot()}), flush=True)
    prefill_seconds = time.monotonic() - started
    generated = []
    for index in range(args.max_tokens):
        token = last["token"]
        if token == 200006:
            break
        generated.append(token)
        if index + 1 < args.max_tokens:
            last = forward([token], True)
    return {"text": tokenizer.decode(generated, skip_special_tokens=True),
            "prompt_tokens": len(ids), "completion_tokens": len(generated), "token_ids": generated,
            "prefill_seconds": prefill_seconds, "total_seconds": time.monotonic()-started,
            "rank0_memory": memory_snapshot(), "rank1_memory": last.get("memory"),
            "checkpoint_sha256": kernels.digest(args.checkpoint / "checkpoint.json"),
            "kv": "scaled FP8 E4M3", "purpose": "experimental text trial"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,default=Path("/model"))
    p.add_argument("--rank",type=int,choices=(0,1),required=True)
    p.add_argument("--master",required=True)
    p.add_argument("--port",type=int,default=18095)
    p.add_argument("--prompt",default="What is 7 * 6? Reply with only the number.")
    p.add_argument("--prompt-file",type=Path)
    p.add_argument("--max-tokens",type=int,default=64)
    p.add_argument("--max-context",type=int,default=524288)
    p.add_argument("--chunk",type=int,choices=(32,64,128),default=128)
    p.add_argument("--seed",type=int,default=115)
    p.add_argument("--temperature",type=float,default=0)
    p.add_argument("--top-p",type=float,default=.95)
    p.add_argument("--protocol-canary",action="store_true")
    args=p.parse_args()
    if args.prompt_file:args.prompt=args.prompt_file.read_text(encoding="utf-8")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    if not args.protocol_canary:
        if not 1<=args.max_tokens<=4096 or not 1<=args.max_context<=524288 or not 0<args.top_p<=1:raise ValueError("Invalid bounded trial limits")
        assert torch.cuda.device_count()==1 and "GB10" in torch.cuda.get_device_name()
        torch.cuda.set_per_process_memory_fraction(.90)
        reader=full.Checkpoint(args.checkpoint)
        if not reader.receipt.get("initial_trial_ready"):raise RuntimeError("Checkpoint did not pass its conversion quality gate")
        rank=json.loads((args.checkpoint/f"rank-{args.rank}.json").read_text())
        assert rank["checkpoint_sha256"]==kernels.digest(args.checkpoint/"checkpoint.json")
        for name,receipt in rank["files"].items():verify_rank_file(args.checkpoint/name,receipt)
        print(json.dumps({"rank":args.rank,"load_budget":check_load_budget(rank,args.max_context,args.chunk)}),flush=True)
    dist.init_process_group("gloo",init_method=f"tcp://{args.master}:{args.port}",rank=args.rank,world_size=2,timeout=timedelta(hours=2))
    try:
        if args.protocol_canary:
            print(json.dumps(protocol_canary(args.rank)),flush=True);return
        stage=Stage(args.checkpoint,args.rank,args.max_context);dist.barrier()
        if args.rank:worker(stage)
        else:
            try:print(json.dumps(generate(stage,args),ensure_ascii=False),flush=True)
            finally:send_json({"command":"stop"},1)
    finally:dist.destroy_process_group()

if __name__=="__main__":main()
