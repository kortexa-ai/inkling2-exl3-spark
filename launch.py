#!/usr/bin/env python3
"""Prepare, download and launch a two-Spark CLI with SSH and Docker."""
import argparse,json,os,pathlib,re,shlex,signal,socket,subprocess,sys,threading,time
ROOT=pathlib.Path(__file__).resolve().parent
STATE=ROOT/'.runtime/setup.json'
REPO='https://github.com/kortexa-ai/inkling2-exl3-spark.git'
NAMES=['inkling2-exl3-rank0','inkling2-exl3-rank1']


def command(peer,args):
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15',peer,shlex.join([str(x) for x in args])] if peer else [str(x) for x in args]


def run(peer,args,**kwargs):
    return subprocess.run(command(peer,args),check=True,**kwargs)


def output(peer,args):
    return run(peer,args,capture_output=True,text=True).stdout.strip()


def memory(peer):
    return int(output(peer,['awk','/MemAvailable/ {print $2}','/proc/meminfo']))*1024


def gpu(peer):
    rows=output(peer,['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader']).splitlines()
    devices=[line.split(',')[0].strip() for line in rows if 'GB10' in line]
    if len(devices)!=1:raise RuntimeError('Expected exactly one NVIDIA GB10 GPU on each host')
    processes=output(peer,['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'])
    if any(devices[0] in line for line in processes.splitlines()):
        raise RuntimeError(f'{peer or "head"}: GPU is busy; stop its model before launching')
    return devices[0]


def setup(peer):
    revision=output(None,['git','-C',ROOT,'rev-parse','HEAD'])
    if output(None,['git','-C',ROOT,'status','--porcelain']):raise RuntimeError('Commit local recipe changes before setup')
    peer_home=output(peer,['sh','-c','printf %s "$HOME"'])
    peer_root=peer_home+'/src/inkling2-exl3-spark'
    exists=subprocess.run(command(peer,['test','-d',peer_root+'/.git']),capture_output=True)
    if exists.returncode:
        run(peer,['git','clone',REPO,peer_root])
    if output(peer,['git','-C',peer_root,'status','--porcelain']):raise RuntimeError('Peer recipe checkout has local changes')
    if output(peer,['git','-C',peer_root,'rev-parse','HEAD'])!=revision:
        run(peer,['git','-C',peer_root,'fetch','origin','main'])
        run(peer,['git','-C',peer_root,'merge','--ff-only','origin/main'])
    if output(peer,['git','-C',peer_root,'rev-parse','HEAD'])!=revision:raise RuntimeError('Peer checkout differs; update both clones to the same commit')
    for host in (None,peer):
        if output(host,['uname','-m'])!='aarch64':raise RuntimeError('Build and run on ARM64 DGX Sparks')
        gpu(host)
        if memory(host)<24*1024**3:raise RuntimeError('Setup needs 24 GiB available RAM; stop the resident model first')
        output(host,['docker','info','--format','{{.ServerVersion}}'])
    image='inkling2-exl3:'+revision[:12]
    run(None,['docker','build','--memory=12g','--shm-size=1g','--tag',image,ROOT])
    image_id=output(None,['docker','image','inspect',image,'--format','{{.Id}}'])
    sender=subprocess.Popen(['docker','image','save',image],stdout=subprocess.PIPE)
    try:
        receiver=run(peer,['docker','image','load'],stdin=sender.stdout)
    finally:
        sender.stdout.close()
        if sender.wait()!=0:raise RuntimeError('Image transfer failed')
    if output(peer,['docker','image','inspect',image,'--format','{{.Id}}'])!=image_id:raise RuntimeError('Image IDs differ')
    state={'peer':peer,'peer_root':peer_root,'image':image,'image_id':image_id,'revision':revision,
           'directories':[str(pathlib.Path.home()/'models/inkling2-exl3'),peer_home+'/models/inkling2-exl3']}
    STATE.parent.mkdir(exist_ok=True);STATE.write_text(json.dumps(state,indent=2)+'\n')
    print('Setup complete. Run ./download.sh '+peer,flush=True)


def load_state(peer):
    state=json.loads(STATE.read_text())
    if state['peer']!=peer:raise RuntimeError('Peer differs from setup; rerun setup.sh')
    if output(None,['git','-C',ROOT,'rev-parse','HEAD'])!=state['revision']:raise RuntimeError('Recipe changed; rerun setup.sh')
    return state


def download(peer):
    state=load_state(peer)
    jobs=[subprocess.Popen(command(host,['python3',str(ROOT/'download.py') if rank==0 else state['peer_root']+'/download.py',
                '--rank',str(rank),'--directory',state['directories'][rank]])) for rank,host in enumerate((None,peer))]
    try:
        results=[p.wait() for p in jobs]
        if any(results):raise RuntimeError('A model download failed; rerun to resume verified files')
    finally:
        for p in jobs:
            if p.poll() is None:p.terminate()


def route(peer,address):
    return json.loads(output(peer,['ip','-j','route','get',address]))[0]


def launch(peer,args):
    state=load_state(peer)
    address=args.head_address or route(None,socket.gethostbyname(peer.rsplit('@',1)[-1]))['prefsrc']
    interfaces=[route(None,socket.gethostbyname(peer.rsplit('@',1)[-1]))['dev'],route(peer,address)['dev']]
    devices=[gpu(host) for host in (None,peer)]
    for host in (None,peer):
        if output(host,['docker','image','inspect',state['image'],'--format','{{.Id}}'])!=state['image_id']:raise RuntimeError('Runtime image changed')
        if output(host,['ss','-ltnH',f'( sport = :{args.port} )']):raise RuntimeError('Rendezvous port is already in use')
        for name in NAMES:
            if subprocess.run(command(host,['docker','container','inspect',name]),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
                raise RuntimeError('An Inkling container already exists; inspect it before starting another')
    identifiers={};streams=[];threads=[];started=time.monotonic()
    def stop_signal(*unused):raise KeyboardInterrupt
    for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(signum,stop_signal)
    logs=ROOT/'.runtime/logs'/time.strftime('%Y%m%d-%H%M%S');logs.mkdir(parents=True)
    try:
        for rank,host in reversed(list(enumerate((None,peer)))):
            cmd=['docker','run','--detach','--name',NAMES[rank],'--gpus','device='+devices[rank],
                 '--network','host','--ipc','host','--ulimit','memlock=-1:-1',
                 '--env','GLOO_SOCKET_IFNAME='+interfaces[rank],
                 '--mount',f'type=bind,src={state["directories"][rank]},dst=/model,readonly',
                 '--mount',f'type=volume,src=inkling2-triton-rank{rank},dst=/root/.triton']
            if rank==0 and args.prompt_file:
                cmd+=['--mount',f'type=bind,src={args.prompt_file.expanduser().resolve()},dst=/prompt.txt,readonly']
            cmd+=[state['image'],'--rank',str(rank),'--master',address,'--port',str(args.port),
                  '--max-context',str(args.max_context),'--max-tokens',str(args.max_tokens),'--chunk',str(args.chunk),
                  '--temperature',str(args.temperature),'--top-p',str(args.top_p)]
            if args.protocol_canary:cmd+=['--protocol-canary']
            if rank==0:cmd+=['--prompt-file','/prompt.txt'] if args.prompt_file else ['--prompt',args.prompt]
            identifier=output(host,cmd)
            if not re.fullmatch('[a-f0-9]{64}',identifier):raise RuntimeError('Invalid container receipt')
            identifiers[rank]=identifier
            stream=subprocess.Popen(command(host,['docker','logs','--follow',identifier]),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            streams.append(stream)
            def copy_log(process,number):
                with (logs/f'rank-{number}.log').open('wb') as target:
                    for line in iter(process.stdout.readline,b''):
                        target.write(line);target.flush()
                        if number==0:sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
            thread=threading.Thread(target=copy_log,args=(stream,rank),daemon=True);thread.start();threads.append(thread)
        while True:
            statuses=[]
            for rank,host in enumerate((None,peer)):
                info=json.loads(output(host,['docker','inspect','--format','{{json .State}}',identifiers[rank]]))
                statuses.append(info)
                if not info['Running'] and info['ExitCode']!=0:raise RuntimeError(f'Rank {rank} failed; see {logs}')
                if memory(host)<6.5*1024**3:raise RuntimeError(f'Rank {rank} crossed the 6.5 GiB host-memory guard')
            if all(not info['Running'] for info in statuses):break
            if time.monotonic()-started>args.timeout:raise TimeoutError('Owned pair reached its time limit')
            time.sleep(2)
        for thread in threads:thread.join(timeout=10)
        print(f'Both ranks exited successfully. Logs: {logs}',flush=True)
    finally:
        for rank,identifier in identifiers.items():
            host=None if rank==0 else peer
            subprocess.run(command(host,['docker','stop','--time','20',identifier]),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            subprocess.run(command(host,['docker','rm',identifier]),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for process in streams:
            if process.poll() is None:process.terminate()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('setup','download','start'));p.add_argument('peer',help='SSH hostname or user@hostname of the second Spark')
    p.add_argument('--prompt',default='What is 7 * 6? Reply with only the number.')
    p.add_argument('--prompt-file',type=pathlib.Path)
    p.add_argument('--max-context',type=int,default=524288);p.add_argument('--max-tokens',type=int,default=64)
    p.add_argument('--chunk',type=int,choices=(32,64,128),default=128)
    p.add_argument('--temperature',type=float,default=0);p.add_argument('--top-p',type=float,default=.95)
    p.add_argument('--port',type=int,default=18095);p.add_argument('--head-address',help='Head IPv4 address reachable from the peer')
    p.add_argument('--timeout',type=int,default=86400,help='Owned launch time limit, in seconds')
    p.add_argument('--protocol-canary',action='store_true')
    args=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.@:-]*',args.peer):p.error('Invalid SSH peer')
    if not 1<=args.max_context<=524288 or not 1<=args.max_tokens<=4096 or not 1<=args.port<=65535 or args.timeout<1:p.error('Invalid launch limit')
    if args.prompt_file and not args.prompt_file.expanduser().is_file():p.error('Prompt file does not exist')
    if args.action=='setup':setup(args.peer)
    elif args.action=='download':download(args.peer)
    else:launch(args.peer,args)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:raise SystemExit(130)
