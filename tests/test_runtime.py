"""CPU checks: output precision, cache continuity and verified downloads."""
import hashlib,io,json,pathlib,sys,tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import model,attention,download,launch,runtime


class RuntimeTests(unittest.TestCase):
    def test_non_thinking_prompt_opens_text_channel(self):
        # The actual pinned template does not act on add_generation_prompt.
        tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**k:'<|message_user|><|content_text|>question<|end_message|>',
                                  encode=lambda text,**k:text)
        self.assertTrue(runtime.encode_prompt(tokenizer,'question').endswith(
            '<|end_message|><|message_model|><|content_text|>'))

    def test_output_head_stays_fp32(self):
        torch.manual_seed(116)
        weights={'model.llm.attn_res.output_query':torch.zeros(32),
                 'model.llm.norm.weight':torch.ones(32).bfloat16(),
                 'model.llm.unembed.weight':torch.randn(7,32).bfloat16()}
        reader=SimpleNamespace(config={'unpadded_vocab_size':7},tensor=lambda key,device:weights[key])
        state=model.Residual(torch.randn(3,32).bfloat16())
        mixed=model.ref.rmsnorm(state.partial,dtype=torch.float32)
        hidden=model.ref.rmsnorm(mixed,weights['model.llm.norm.weight'])
        raw=hidden.float()@weights['model.llm.unembed.weight'].float().T
        torch.testing.assert_close(model.logits(reader,state),30*(raw/30).tanh(),rtol=2e-6,atol=2e-6)

    def test_cache_appends_without_reallocating_global_storage(self):
        cache={'capacity':5}
        value=lambda start,n:(torch.arange(start,start+n).reshape(n,1,1),torch.ones(n,1,1))
        attention.append_cache(cache,value(0,3),value(10,3),local=False)
        ptr=cache['k'][0].data_ptr()
        k,v,offset=attention.append_cache(cache,value(3,2),value(13,2),local=False)
        self.assertEqual(ptr,k[0].data_ptr());self.assertEqual(offset,0)
        torch.testing.assert_close(k[0].flatten(),torch.arange(5))
        with self.assertRaises(ValueError):attention.append_cache(cache,value(5,1),value(15,1),local=False)

    def test_download_requires_size_and_hash_before_replacing(self):
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'weights';path.write_bytes(b'old')
            data=b'new verified weights';sha=hashlib.sha256(data).hexdigest()
            with patch('urllib.request.urlopen',return_value=io.BytesIO(b'corrupt')):
                with self.assertRaises(ValueError):download.download('weights',path,sha,len(data))
            self.assertEqual(path.read_bytes(),b'old')
            with patch('urllib.request.urlopen',return_value=io.BytesIO(data)):
                download.download('weights',path,sha,len(data))
            self.assertEqual(path.read_bytes(),data)
            with patch('urllib.request.urlopen',side_effect=AssertionError('must use verified local file')):
                download.download('weights',path,sha,len(data))

    def test_artifact_paths_cannot_escape_download_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory)
            for name in ('../escape','/absolute'):
                with self.assertRaises(ValueError):download.safe_path(root,name)
            (root/'link').symlink_to('/tmp')
            with self.assertRaises(ValueError):download.safe_path(root,'link/escape')

    def test_remote_commands_quote_prompt_as_data(self):
        args=['echo','hello; $(false) `false`']
        cmd=launch.command('peer',args)
        import shlex
        self.assertEqual(shlex.split(cmd[-1]),args)

    def test_network_selection_resolves_ssh_alias_hostname(self):
        with patch.object(launch,'output',return_value='hostname 192.0.2.17\nuser spark') as config:
            self.assertEqual(launch.peer_address('spark-alias'),'192.0.2.17')
        config.assert_called_once_with(None,['ssh','-G','spark-alias'])



@unittest.skipUnless(__import__('os').environ.get('INKLING_GPU_TEST')=='1','opt-in GPU conformance')
class GPUConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert torch.cuda.device_count()==1
        torch.cuda.set_per_process_memory_fraction(.05)
        torch.backends.cuda.matmul.allow_tf32=False

    def test_attention_reference_and_512k_capacity(self):
        import math_ops,time
        torch.manual_seed(115)
        for n,m,offset,window in [(1,1,0,None),(129,32,97,None),(1151,128,1023,1023),(4096,1,4095,None),(8192,32,8160,None)]:
            q=torch.randn(m,48,128,device='cuda',dtype=torch.bfloat16)
            k=model.pack_kv(torch.randn(n,8,128,device='cuda',dtype=torch.bfloat16))
            v=model.pack_kv(torch.randn(n,8,128,device='cuda',dtype=torch.bfloat16))
            r=torch.randn(m,48,1024,device='cuda',dtype=torch.bfloat16)*.2
            actual,lse=attention.attention(q,k,v,r,offset,0,window)
            expected,expected_lse=math_ops.attention(q,model.unpack_kv(k),model.unpack_kv(v),r,
                torch.arange(offset,offset+m,device='cuda'),torch.arange(n,device='cuda'),window)
            self.assertLess(float((actual-expected).norm()/expected.norm()),1e-4)
            torch.testing.assert_close(lse,expected_lse,rtol=0,atol=2e-5)
            del q,k,v,r,actual,lse,expected,expected_lse
            torch.cuda.empty_cache()
        capacity=524288;caches=[]
        for unused in range(4):
            k=torch.empty((capacity,8,128),device='cuda',dtype=torch.float8_e4m3fn).fill_(1.)
            v=torch.empty_like(k).fill_(2.)
            scale=torch.full((capacity,8,1),.125,device='cuda')
            caches.append(((k,scale),(v,scale.clone())))
        for m in (1,128):
            q=torch.full((m,48,128),.125,device='cuda',dtype=torch.bfloat16)
            r=torch.zeros((m,48,1024),device='cuda',dtype=torch.bfloat16)
            torch.cuda.synchronize();started=time.monotonic()
            actual,lse=attention.attention(q,*caches[0],r,capacity-m)
            torch.cuda.synchronize()
            expected_lse=torch.arange(capacity-m+1,capacity+1,device='cuda').float().log()+128*.125*.125/128**.5
            torch.testing.assert_close(actual,torch.full_like(actual,.25),rtol=0,atol=1e-4)
            torch.testing.assert_close(lse,expected_lse[:,None].expand_as(lse),rtol=0,atol=1e-4)
            print(json.dumps({'context':capacity,'rows':m,'seconds':time.monotonic()-started,
                              'peak_gib':torch.cuda.max_memory_allocated()/1024**3}),flush=True)
        del caches,k,v,scale,q,r,actual,lse
        torch.cuda.empty_cache()

    @unittest.skipUnless(__import__('os').environ.get('INKLING_FIXTURES'),'optional independent codec fixture directory')
    def test_codec_against_independent_fixtures(self):
        import os,kernels
        from safetensors.torch import load_file
        from exllamav3.modules.quant.exl3 import LinearEXL3
        root=pathlib.Path(os.environ['INKLING_FIXTURES']);ops=model.Ops()
        for item in json.loads((root/'manifest.json').read_text())['fixtures']:
            path=download.safe_path(root,item['file'])
            self.assertEqual(kernels.digest(path),item['sha256'])
            data=load_file(path)
            linear=LinearEXL3(None,item['width'],4608,**{k:data[k].cuda() for k in ('suh','svh','trellis','mul1')})
            self.assertEqual(linear.K,item['K'])
            for rows in (1,2,4,32,128):
                x=data['x'].cuda().repeat((rows+31)//32,1)[:rows]
                expected=data['reference'].cuda().repeat((rows+31)//32,1)[:rows]
                result=linear.forward(x,{},out_dtype=torch.float32)
                error=float((result-expected).norm()/expected.norm())
                self.assertLess(error,.003)
                print(json.dumps({'K':item['K'],'projection':item['projection'],'rows':rows,'nrmse':error}),flush=True)
            raw=data['raw'].cuda();packed,scales,amax=ops.quant(raw)
            self.assertTrue(torch.equal(packed.cpu(),data['a4_packed']))
            self.assertTrue(torch.equal(scales.view(torch.uint8).cpu(),data['a4_scales']))
            self.assertTrue(torch.equal(amax.cpu(),data['a4_amax']))
            mx,ms=ops.mxfp8(raw)
            self.assertTrue(torch.equal(mx.view(torch.uint8).cpu(),data['mx_values']))
            self.assertTrue(torch.equal(ms.cpu(),data['mx_scales']))
            del linear,data,x,expected,result,raw,packed,scales,amax,mx,ms
            torch.cuda.empty_cache()

if __name__=='__main__':unittest.main()
