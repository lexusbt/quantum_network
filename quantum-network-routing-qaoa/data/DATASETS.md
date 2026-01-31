# Dataset Inventory

## Raw SNAP Datasets (Not in Git)

The following datasets should be downloaded and placed in `data/raw/`:

### Available Datasets

| Dataset | Filename | Size | Nodes | Edges | Description |
|---------|----------|------|-------|-------|-------------|
| ca-GrQc | ca-GrQc.txt | 156 KB | 5,242 | 14,496 | General Relativity collaboration network |
| email-Enron | email-Enron.txt | 4.5 MB | 36,692 | 183,831 | Email communication network |
| p2p-Gnutella08 | p2p-Gnutella08.txt | 165 KB | 6,301 | 20,777 | Gnutella P2P network |
| ca-CondMat | ca-CondMat.txt | 1.9 MB | 23,133 | 93,497 | Condensed matter collaboration |
| ca-HepPh | ca-HepPh.txt | 2.5 MB | 12,008 | 118,521 | High energy physics collaboration |

## How to Get the Datasets

### Option 1: Manual Download
Download from SNAP: https://snap.stanford.edu/data/

1. ca-GrQc: https://snap.stanford.edu/data/ca-GrQc.txt.gz
2. email-Enron: https://snap.stanford.edu/data/email-Enron.txt.gz
3. p2p-Gnutella08: https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz
4. ca-CondMat: https://snap.stanford.edu/data/ca-CondMat.txt.gz
5. ca-HepPh: https://snap.stanford.edu/data/ca-HepPh.txt.gz

Extract all `.gz` files to `data/raw/`

### Option 2: Use Download Script (Future)
```bash
python scripts/download_datasets.py
```

## Current Status

✅ **These datasets are currently available locally in `data/raw/`**

Last updated: 2026-01-28
