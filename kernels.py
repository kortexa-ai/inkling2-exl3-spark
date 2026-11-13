"""Audited source kernels; upstream files are pinned and fetched at image build."""
import ast,hashlib,math,sys,os
from pathlib import Path
from types import ModuleType
from dataclasses import dataclass,field
import torch,triton
import triton.language as tl
from triton.language.extra import libdevice

GIB=1024**3
SG="sglang/python/sglang/"
CACHE=Path(__file__).resolve().parent/"upstream"

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024**2):
            h.update(block)
    return h.hexdigest()

@triton.jit
def decode_fp4(Packed, Scales, Outer, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    packed = tl.load(Packed + i // 2, i < N, other=0).to(tl.int32)
    code = (packed >> ((i % 2) * 4)) & 15
    magnitude = code & 7
    exponent = tl.maximum(magnitude >> 1, 1) - 1
    significand = (magnitude & 1) + tl.where(magnitude >= 2, 2, 0)
    value = significand.to(tl.float32) * 0.5 * (1 << exponent).to(tl.float32)
    value = tl.where((code & 8) != 0, -value, value)
    scale_bits = tl.load(Scales + i // 16, i < N, other=0).to(tl.uint8)
    scale = scale_bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    outer = tl.load(Outer)
    tl.store(Out + i, (value * scale) * outer, i < N)

def decode(packed, scales, outer, output):
    decode_fp4[(triton.cdiv(output.numel(), 1024),)](
        packed, scales.view(torch.uint8), outer, output, output.numel(), 1024,
        enable_fp_fusion=False)

def extract(name, selected, extra=None, method=None):
    """Execute only audited AST definitions, with no upstream import side effects."""
    path = CACHE / name
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in selected]
    if method:
        nodes = [node for node in nodes[0].body if isinstance(node, ast.FunctionDef) and node.name == method]
    if len(nodes) != (1 if method else len(selected)):
        raise ValueError(f"Missing audited definitions: {name}: {selected}")
    module = ModuleType("inkling_oracle_" + hashlib.sha256((name + str(selected)).encode()).hexdigest()[:12])
    module.__file__ = str(path)
    module.__dict__.update(torch=torch, triton=triton, tl=tl, libdevice=libdevice,
                           math=math, dataclass=dataclass, field=field, PDL=False)
    module.__dict__.update(extra or {})
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module

class Oracles:
    def __init__(self):
        self.quant=extract(SG+"kernels/ops/inkling2/nvfp4.py",["_pack_roundtrip","_block_error","_row_amax","_quantize_rows","_quantize","_quantize_chunk","_quantize_values","nvfp4_quantize"]).nvfp4_quantize
        helpers=extract(SG+"kernels/ops/moe/gate_topk.py",["get_topmask_and_fullmask","fpval_to_key","key_to_indx"])
        self.router=extract(SG+"kernels/ops/inkling2/route.py",["_route_topk","_pick","_logsigmoid_renorm8","route_topk"],{k:getattr(helpers,k) for k in ("fpval_to_key","key_to_indx")}).route_topk
        self.swiglu=extract(SG+"kernels/ops/inkling2/moe.py",["_swiglu","_capped_swiglu","swiglu"]).swiglu
        self.sconv=extract(SG+"kernels/ops/inkling2/sconv.py",["_sconv_kernel","_sconv_prefill_kernel","short_convolution"],{"PREFILL_MIN_ROWS":2048,"_cuda_sconv":lambda *a,**k:False}).short_convolution

    def route(self, logits, bias):
        return self.router(logits[None], bias, 8, route_scale=8., global_scale=logits.new_ones(()))

    def conv(self, x, weight, history=None):
        cache = x.new_zeros((2, 3, x.shape[-1]))
        if history is not None:
            cache[1] = history
        metadata = {"safe_idx": torch.ones(1, dtype=torch.long, device=x.device),
                    "cache_mask": torch.ones((1, 1, 1), dtype=torch.bool, device=x.device),
                    "cu": torch.tensor([0, len(x)], dtype=torch.long, device=x.device),
                    "si": torch.zeros(len(x), dtype=torch.int32, device=x.device)}
        out = self.sconv(x.contiguous(), weight.reshape(x.shape[-1], 4).contiguous(), cache, metadata)
        raw = torch.cat((cache[1], x))
        return out, raw[-3:].clone()

def discard_file_cache(path):
    # These immutable checkpoint pages are no longer needed after a CUDA copy.
    # Avoid a host-wide cache flush or a privileged launcher on unified memory.
    with path.open("rb") as source:
        os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

