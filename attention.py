"""Bounded-workspace GQA with Inkling relative bias and scaled FP8 KV."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["NK", "Q0", "K0"])
def _attention(Q, K, KS, V, VS, R, O, LSE,
               NQ:tl.constexpr, NK, H:tl.constexpr, HK:tl.constexpr,
               D:tl.constexpr, Q0, K0,
               WINDOW:tl.constexpr, BM:tl.constexpr, BN:tl.constexpr):
    head=tl.program_id(1)
    kh=head//(H//HK)
    qi=tl.program_id(0)*BM+tl.arange(0,BM)
    di=tl.arange(0,D)
    ni=tl.arange(0,BN)
    q=tl.load(Q+(qi[:,None]*H+head)*D+di[None,:],qi[:,None]<NQ,0)
    maximum=tl.full((BM,),float('-inf'),tl.float32)
    denominator=tl.zeros((BM,),tl.float32)
    accumulator=tl.zeros((BM,D),tl.float32)
    for start in range(tl.cdiv(NK,BN)):
        key=start*BN+ni
        kb=tl.load(K+(key[None,:]*HK+kh)*D+di[:,None],key[None,:]<NK,0.).to(tl.float32)
        scale=tl.load(KS+key*HK+kh,key<NK,0)
        # Match the FP8-cache decode's BF16 rounding boundary.
        decoded=(kb*scale[None,:]).to(tl.bfloat16)
        score=tl.dot(q,decoded)*D**-0.5
        distance=Q0+qi[:,None]-(K0+key[None,:])
        relative=tl.load(R+(qi[:,None]*H+head)*1024+tl.minimum(tl.maximum(distance,0),1023),
                         (qi[:,None]<NQ)&(distance>=0)&(distance<1024),0).to(tl.float32)
        score=score+relative
        valid=(qi[:,None]<NQ)&(key[None,:]<NK)&(distance>=0)
        if WINDOW:
            valid=valid&(distance<=WINDOW)
        score=tl.where(valid,score,float('-inf'))
        next_max=tl.maximum(maximum,tl.max(score,1))
        alpha=tl.where(maximum==float('-inf'),0.,tl.exp(maximum-next_max))
        probability=tl.where(valid,tl.exp(score-next_max[:,None]),0.)
        vb=tl.load(V+(key[:,None]*HK+kh)*D+di[None,:],key[:,None]<NK,0.).to(tl.float32)
        vscale=tl.load(VS+key*HK+kh,key<NK,0)
        decoded_v=(vb*vscale[:,None]).to(tl.bfloat16).to(tl.float32)
        accumulator=accumulator*alpha[:,None]+tl.dot(probability,decoded_v,input_precision='tf32x3')
        denominator=denominator*alpha+tl.sum(probability,1)
        maximum=next_max
    result=accumulator/denominator[:,None]
    tl.store(O+(qi[:,None]*H+head)*D+di[None,:],result,qi[:,None]<NQ)
    tl.store(LSE+qi*H+head,maximum+tl.log(denominator),qi<NQ)


def attention(q, packed_k, packed_v, relative, query_offset, key_offset=0, window=None):
    rows,heads,width=q.shape
    k,ks=packed_k;v,vs=packed_v
    assert q.dtype==torch.bfloat16 and k.dtype==v.dtype==torch.float8_e4m3fn
    assert width==128 and heads%k.shape[1]==0
    assert k.shape==v.shape and ks.shape==vs.shape
    assert relative.shape==(rows,heads,1024)
    assert all(t.is_contiguous() for t in (q,k,ks,v,vs,relative))
    output=torch.empty(q.shape,device=q.device,dtype=torch.float32)
    lse=torch.empty((rows,heads),device=q.device,dtype=torch.float32)
    block=16 if rows<=16 else 32
    _attention[(triton.cdiv(rows,block),heads)](q,k,ks,v,vs,relative,output,lse,
        rows,len(k),heads,k.shape[1],width,query_offset,key_offset,window or 0,block,64,
        num_warps=4,num_stages=2,enable_fp_fusion=False)
    return output,lse


def append_cache(cache, current_k, current_v, *, local):
    """Append in place globally; retain only a short history for local layers."""
    offset = cache.get('position', 0)
    rows = len(current_k[0])
    if local:
        k = tuple(torch.cat((a, b)) for a, b in zip(cache['k'], current_k)) if 'k' in cache else current_k
        v = tuple(torch.cat((a, b)) for a, b in zip(cache['v'], current_v)) if 'v' in cache else current_v
        key_offset = offset + rows - len(k[0])
        cache['k'] = tuple(t[-1023:].clone() for t in k)
        cache['v'] = tuple(t[-1023:].clone() for t in v)
    else:
        capacity = cache['capacity']
        if offset + rows > capacity:
            raise ValueError('KV cache capacity exceeded')
        if 'k' not in cache:
            for name, current in (('k', current_k), ('v', current_v)):
                cache[name] = tuple(torch.empty((capacity, *t.shape[1:]), dtype=t.dtype, device=t.device)
                                    for t in current)
        for name, current in (('k', current_k), ('v', current_v)):
            for storage, value in zip(cache[name], current):
                storage[offset:offset+rows].copy_(value)
        k = tuple(t[:offset+rows] for t in cache['k'])
        v = tuple(t[:offset+rows] for t in cache['v'])
        key_offset = 0
    cache['position'] = offset + rows
    return k, v, key_offset


def layer_attention(layer, sources, cache, *, pack_kv, ref):
    """Keep the qualified projection/normalization/gate math around tiled GQA."""
    w = layer.weights
    qsource, ksource, vsource = sources * 3 if len(sources) == 1 else sources
    q = ref.linear(qsource, w['attn.wq_du.weight']).view(-1, 48, 128)
    k = layer.convolve(ref.linear(ksource, w['attn.wk_dv.weight']), 'attn.k_sconv', cache,
                       None if layer.local else ksource).view(-1, 8, 128)
    v = layer.convolve(ref.linear(vsource, w['attn.wv_dv.weight']), 'attn.v_sconv', cache,
                       None if layer.local else vsource).view(-1, 8, 128)
    r = ref.linear(qsource, w['attn.wr_du.weight']).view(-1, 48, 16)
    q = ref.l2clip(q, w['attn.q_l2_clip.scale'])
    k, v = ref.l2clip(k), ref.rmsnorm(v)
    r = ref.l2clip(r, w['attn.rel_l2_clip.scale'], 8)
    offset = cache.get('position', 0)
    if not layer.local:
        positions = torch.arange(offset, offset+len(q), device=q.device)
        tau = 1 + .1 * torch.log(((positions+1).float()/128000).clamp_min(1))
        q = (q.float()*tau[:, None, None]).bfloat16()
        r = (r.float()*tau[:, None, None]).bfloat16()
    relative = ref.linear(r.reshape(-1, 16), w['attn.rel_logits_proj.proj'].T).view(-1, 48, 1024)
    packed_k, packed_v, key_offset = append_cache(cache, pack_kv(k), pack_kv(v), local=layer.local)
    output, lse = attention(q.contiguous(), packed_k, packed_v, relative.contiguous(), offset,
                            key_offset, 1023 if layer.local else None)
    gate = ref.linear(qsource, w['attn.gate_attn.weight']).view(-1, 48, 128).float()
    sink = ref.linear(qsource, w['attn.attn_sink_proj.weight']).float()
    output = (output*gate.sigmoid()*(lse-sink).sigmoid()[..., None]).bfloat16()
    return ref.linear(output.flatten(1), w['attn.wo_ud.weight'])
