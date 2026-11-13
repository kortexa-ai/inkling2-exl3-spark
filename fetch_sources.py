"""Fetch only pinned, hash-verified public build inputs (Python standard library)."""
import hashlib,json,pathlib,tarfile,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent


def fetch(url, target, expected):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:return
    with urllib.request.urlopen(url,timeout=120) as response:
        data=response.read()
    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError(f'Source checksum mismatch: {target.name}')
    target.write_bytes(data)


def main():
    for name,pin in json.loads((ROOT/'source-pins.json').read_text()).items():
        fetch(pin['url'],ROOT/'upstream'/name,pin['sha256'])
    archive=ROOT/'upstream/build/exllamav3.tar.gz'
    destination=ROOT/'upstream/build/exllamav3'
    if not destination.exists():
        with tarfile.open(archive) as source:source.extractall(destination.parent,filter='data')
        (destination.parent/'exllamav3-151539c77abc7ab7425d30da7a4e8e3c5c154e7b').rename(destination)

if __name__=='__main__':main()
