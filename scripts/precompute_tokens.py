import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NET_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
KMER_BERT_PATH = os.path.join(_NET_ROOT, "DNA_bert_6")
DNABERT2_PATH  = os.path.join(_NET_ROOT, "DNABERT-2-117M")


def load_seq_label(path):
    sequences, labels = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            sequences.append(parts[1])
            labels.append(int(parts[2]))
    return sequences, labels


def kmer_split(seqs, k):
    out = []
    for s in seqs:
        pieces = [s[x:x + k] for x in range(len(s) + 1 - k)]
        out.append(" ".join(pieces))
    return out


def tokenize_split(sequences, labels, kmer_tokenizer, dnabert2_tokenizer, kmer_k):

    # ---- k-mer BERT branch ----
    kmer_strings = kmer_split(sequences, kmer_k)
    kmer_enc = kmer_tokenizer(kmer_strings, return_tensors='pt',
                              padding=True, truncation=True)

    # ---- DNABERT2 branch ----
    d2_enc = dnabert2_tokenizer(list(sequences), return_tensors='pt',
                                padding=True, truncation=True)

    return {
        'labels'             : torch.tensor(labels, dtype=torch.long),
        'kmer_input_ids'     : kmer_enc['input_ids'],
        'kmer_attn_mask'     : kmer_enc['attention_mask'],
        'kmer_tok_type'      : kmer_enc['token_type_ids'],
        'dnabert2_input_ids' : d2_enc['input_ids'],
        'sequences'          : list(sequences),
    }


def process_dataset(ds_name, datasets_root, cache_root,
                    kmer_tokenizer, dnabert2_tokenizer, kmer_k):
    src = os.path.join(datasets_root, ds_name)
    dst = os.path.join(cache_root, ds_name)
    os.makedirs(dst, exist_ok=True)

    for split in ('train', 'test'):
        in_path  = os.path.join(src, f"{split}.data")
        out_path = os.path.join(dst, f"{split}.pt")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists")
            continue
        if not os.path.exists(in_path):
            print(f"[warn] missing {in_path}")
            continue

        seqs, labels = load_seq_label(in_path)
        pack = tokenize_split(seqs, labels, kmer_tokenizer, dnabert2_tokenizer, kmer_k)
        torch.save(pack, out_path)
        print(f"[done] {out_path}  N={len(labels)}  "
              f"kmer_L={pack['kmer_input_ids'].shape[1]}  "
              f"d2_L={pack['dnabert2_input_ids'].shape[1]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_root", type=str,
                        default=os.path.abspath(os.path.join(
                            _THIS_DIR, "..", "Datasets_dfm")))
    parser.add_argument("--cache_root", type=str,
                        default=os.path.abspath(os.path.join(
                            _THIS_DIR, "..", "cache_tokens_dfm")),
                        help="Where to write per-dataset .pt caches")
    parser.add_argument("--kmer_k", type=int, default=6)
    parser.add_argument("--single_dataset", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.cache_root, exist_ok=True)
    print("Loading tokenizers...")
    kmer_tok = BertTokenizer.from_pretrained(KMER_BERT_PATH)
    d2_tok   = AutoTokenizer.from_pretrained(DNABERT2_PATH, trust_remote_code=True)

    if args.single_dataset:
        ds_list = [args.single_dataset]
    else:
        with open(os.path.join(args.datasets_root, "FILESLIST.txt"), 'r') as f:
            ds_list = [line.strip().replace('\r', '') for line in f if line.strip()]

    print(f"Datasets to tokenize: {len(ds_list)}")
    for ds in tqdm(ds_list):
        process_dataset(ds, args.datasets_root, args.cache_root,
                        kmer_tok, d2_tok, args.kmer_k)


if __name__ == "__main__":
    main()
