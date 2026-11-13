"""48-layer Inkling EXL3 text graph; native activation quantization is retained."""
import json,os
from pathlib import Path
from dataclasses import dataclass,field
import torch,triton
from safetensors import safe_open
import kernels
import math_ops as ref
os.environ["EXL3_GEMV"]="0"
os.environ["EXL3_INT8_GEMV"]="0"

def mixture(sources, query):
    """Depth mixture stays FP32, including an unrounded partial+attention sum."""
    values = torch.stack([x.float() for x in sources])
    inverse = torch.rsqrt(values.square().mean(-1) + 1e-6)
    query = query.float().reshape(-1, values.shape[-1]) * values.shape[-1]**-.5
    scores = torch.einsum("sth,qh->qst", values, query) * inverse[None]
    probabilities = scores.softmax(1)
    return torch.einsum("qst,sth->qth", probabilities, values * inverse[..., None])

@dataclass
class Residual:
    partial: torch.Tensor
    blocks: list[torch.Tensor] = field(default_factory=list)

    def to(self, device):
        return Residual(self.partial.to(device), [b.to(device) for b in self.blocks])

    def attention_inputs(self, layer, query, local):
        if not self.blocks:
            return [self.partial] * (1 if local else 3)
        return list(mixture([*self.blocks, self.partial], query))

    def mlp_input(self, layer, attention, query):
        sources = ([*self.blocks, self.partial, attention] if layer % 6 == 0
                   else [*self.blocks, self.partial.float() + attention.float()])
        return mixture(sources, query)[0]

    def finish(self, layer, attention, mlp):
        if layer % 6 == 0:
            self.blocks.append(self.partial)
            total = attention.float() + mlp.float()
        else:
            total = (self.partial.float() + attention.float()) + mlp.float()
        self.partial = total.bfloat16()

def mxfp8_dequant(values, scales):
    factors = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    return (values.float() * factors).bfloat16()

def hadamard128(x):
    """Independent normalized Sylvester transform in FP32, block size 128."""
    shape = x.shape
    values = x.float().reshape(-1, 128)
    width = 1
    while width < 128:
        pairs = values.reshape(-1, 128//(2*width), 2, width)
        low, high = pairs[:, :, 0], pairs[:, :, 1]
        values = torch.cat((low+high, low-high), -1).reshape(-1,128)
        width *= 2
    return (values * 128**-.5).reshape(shape)

def rotated_linear_reference(x, linear):
    # Native EXL3 first multiplies the input scales in FP16, then stores
    # the input Hadamard in FP16. Reconstructing a full FP16 weight matrix
    # moves those rounding boundaries and is a separate numerical comparison.
    scaled = (x.float() * linear.suh.float()).half()
    transformed = hadamard128(scaled).half()
    output = transformed.float() @ linear.get_inner_weight_tensor().float()
    return hadamard128(output) * linear.svh.float()

class Ops(kernels.Oracles):
    def __init__(self):
        super().__init__()
        self.mx = kernels.extract(kernels.SG + "kernels/ops/inkling2/moe.py", ["_mxfp8_block", "_mxfp8_quantize"])

    def mxfp8(self, x):
        x = x.contiguous()
        values = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scales = torch.empty((*x.shape[:-1], x.shape[-1] // 32), device=x.device, dtype=torch.uint8)
        self.mx._mxfp8_quantize[(x.numel() // x.shape[-1], triton.cdiv(x.shape[-1], 256))](
            x, values, scales, x.shape[-1], 256, enable_fp_fusion=False, launch_pdl=False)
        return values, scales

    def conv_dynamic(self, x, weight, dynamic, history=None):
        cache = x.new_zeros((2, 3, x.shape[-1]))
        if history is not None:
            cache[1] = history
        metadata = {"safe_idx": torch.ones(1, dtype=torch.long, device=x.device),
                    "cache_mask": torch.ones((1, 1, 1), dtype=torch.bool, device=x.device),
                    "cu": torch.tensor([0, len(x)], dtype=torch.long, device=x.device),
                    "si": torch.zeros(len(x), dtype=torch.int32, device=x.device)}
        out = self.sconv(x.contiguous(), weight, cache, metadata, dynamic_weights=dynamic.contiguous())
        return out, torch.cat((cache[1], x))[-3:].clone()

def pack_kv(x):
    scale = (x.float().abs().amax(-1, keepdim=True) / 448).clamp_min(1e-30)
    values = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return values, scale

def unpack_kv(item):
    return (item[0].float() * item[1]).bfloat16()

class Layer:
    def __init__(self, reader, index, ops, *, device="cuda"):
        self.reader, self.index, self.ops, self.device = reader, index, ops, device
        self.local = index in reader.config["local_layer_ids"]
        self.nvfp4 = index < 40
        prefix = f"model.llm.layers.{index}."
        self.weights = {k[len(prefix):]: reader.tensor(k, device) for k in reader.layer_keys(index)}
        self.mlp_query = reader.tensor(f"model.llm.attn_res.mlp_queries.{index}", device)
        self.attn_query = reader.tensor(f"model.llm.attn_res.attn_queries.{index-1}", device) if index else None
        self.bank, self.outer, self.packed = {}, {}, {}

    def activation(self, x):
        if not self.nvfp4:
            return mxfp8_dequant(*self.ops.mxfp8(x)).float()
        packed, scales, amax = self.ops.quant(x.contiguous())
        inner = torch.empty_like(x, dtype=torch.bfloat16)
        kernels.decode(packed, scales, torch.ones(1, device=x.device), inner)
        return inner.float() * (amax[:, None] * (1 / 1536))

    def load_packed(self):
        from exllamav3.modules.quant.exl3 import LinearEXL3
        for name, k, n in (("w13", 4608, 4608), ("w2", 2304, 4608)):
            self.packed[name] = []
            for tensors in self.reader.experts(self.index, name, self.device):
                self.packed[name].append(LinearEXL3(None, k, n, **tensors))
            assert len(self.packed[name]) == 384

    def expert(self, name, expert, x):
        if not self.packed:
            raise RuntimeError("Packed experts have not been loaded")
        # Keep activation quantization even though EXL3's dot consumes FP16.
        values = self.activation(x)
        return self.packed[name][expert].forward(values.half(), {}, out_dtype=torch.float32).bfloat16()

    def convolve(self, x, name, cache, dynamic_input=None):
        history = cache.get(name)
        if dynamic_input is None:
            output, history = self.ops.conv(x, self.weights[name + ".weight"], history)
        else:
            dynamic = ref.linear(dynamic_input, self.weights[name + ".weight_proj"]).view(-1, 8, 4)
            output, history = self.ops.conv_dynamic(x, self.weights[name + ".static_w"], dynamic, history)
        cache[name] = history
        return output

    def attention(self, sources, cache, *, fp8_cache=False):
        if not fp8_cache:
            raise ValueError('Streaming attention requires scaled FP8 KV')
        from attention import layer_attention
        return layer_attention(self, sources, cache, pack_kv=pack_kv, ref=ref)

    def prologue(self, state, cache, *, fp8_cache=False):
        streams = state.attention_inputs(self.index, self.attn_query, self.local)
        names = ["attn_norm"] if self.local else ["q_input_norm", "k_input_norm", "v_input_norm"]
        normalized = [ref.rmsnorm(value, self.weights[name + ".weight"]) for name, value in zip(names, streams)]
        attention = self.attention(normalized, cache, fp8_cache=fp8_cache)
        attention = self.convolve(attention, "attn_sconv", cache)
        x = ref.rmsnorm(state.mlp_input(self.index, attention, self.mlp_query), self.weights["mlp_norm.weight"])
        logits = ref.linear(x, self.weights["mlp.gate.weight"], output_dtype=torch.float32)
        _, ids, gammas = self.ops.route(logits, self.weights["mlp.gate.bias"])
        return attention, x, ids.long(), gammas

    def moe(self, x, ids, gammas):
        rows = torch.empty((len(x), 8, 4608), dtype=torch.bfloat16, device=x.device)
        for expert in ids.unique().tolist():
            token, slot = torch.where(ids == expert)
            gu = self.expert("w13", expert, x[token])
            hidden = self.ops.swiglu(gu, gammas[token, slot], 4, 25)
            rows[token, slot] = self.expert("w2", expert, hidden)
        total = torch.zeros_like(x, dtype=torch.float32)
        for slot in range(8):
            total = total + rows[:, slot].float()
        return total.bfloat16()

    def forward(self, state, cache, *, fp8_cache=False):
        attention, x, ids, gammas = self.prologue(state, cache, fp8_cache=fp8_cache)
        output = self.convolve(self.moe(x, ids, gammas), "mlp_sconv", cache)
        state.finish(self.index, attention, output)


def embed(reader, ids, device="cuda"):
    # Preserve the full embedding precision without a second GPU table copy.
    rows = torch.nn.functional.embedding(torch.tensor(ids), reader.tensor("model.llm.embed.weight"))
    return ref.rmsnorm(rows.to(device), reader.tensor("model.llm.embed_norm.weight", device))

def logits(reader, state, *, positions=None):
    device = state.partial.device
    if positions is not None:
        state = Residual(state.partial[positions], [b[positions] for b in state.blocks])
    final = mixture([*state.blocks, state.partial], reader.tensor("model.llm.attn_res.output_query", device))[0]
    hidden = ref.rmsnorm(final, reader.tensor("model.llm.norm.weight", device))
    # hf_quant_config explicitly requests fwd_output_dtype=float32 for unembed,
    # while retaining BF16 checkpoint weights. Do not round logits through BF16.
    weight = reader.tensor("model.llm.unembed.weight", device)
    values = ref.linear(hidden, weight, output_dtype=torch.float32)
    values = 30 * (values / 30).tanh()
    return values[:, :reader.config["unpadded_vocab_size"]]

class Checkpoint:
    def __init__(self, root, *, require_complete=True):
        self.root = Path(root)
        self.receipt = json.loads((self.root / "checkpoint.json").read_text())
        if require_complete and self.receipt.get("status") != "complete":
            raise RuntimeError("Checkpoint is not complete")
        self.config = self.receipt["text_config"]
        self.index = self.receipt["weight_map"]
        self._globals = None

    def tensor(self, key, device="cpu"):
        with safe_open(self.root / self.index[key], framework="pt", device="cpu") as reader:
            return reader.get_tensor(key).to(device)

    def layer_keys(self, layer):
        prefix = f"model.llm.layers.{layer}."
        return [k for k in self.index if k.startswith(prefix) and ".experts." not in k]

    def experts(self, layer, name, device):
        from safetensors import safe_open
        # Copy only this projection. Loading the whole batch twice interleaves
        # retained weights with temporary copies and fragments the CUDA pool.
        for batch in self.receipt["layers"][str(layer)]["batches"]:
            path = self.root/batch["file"]
            with safe_open(path, framework="pt", device=str(device)) as reader:
                for expert in range(batch["start"], batch["end"]):
                    prefix = f"experts.{expert}.{name}."
                    yield {key: reader.get_tensor(prefix+key)
                           for key in ("suh", "svh", "trellis", "mul1")}
            kernels.discard_file_cache(path)

