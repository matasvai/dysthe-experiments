"""Verify the published compact snapshot; does not execute archived research code."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
for entry in manifest['files']:
 path=root/entry['archive_path']
 assert path.is_relative_to(root) and path.is_file(),path
 assert path.stat().st_size==entry['bytes'],path
 assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],path
 assert path.suffix not in {'.npz','.npy','.pt','.pth','.h5','.hdf5'},path
 assert path.stat().st_size<=5_000_000,path
 if path.suffix=='.json':json.loads(path.read_text())
print(f"Verified {len(manifest['files'])} archived source/evidence files; {len(manifest['omitted_artifacts'])} raw or large artifacts indexed outside Git.")
