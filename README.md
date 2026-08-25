# DV-Prom

Leveraging the fine-grained local representations of DNABERT alongside the semantic-focused tokenization of DNABERT-2, we propose DV-Prom, a Dual-View model architecture that integrates DNABERT 6-mer and DNABERT-2. 

## Workflow 
<img width="7383" height="3542" alt="iPro-BERTs" src="https://github.com/user-attachments/assets/e647f95e-5821-4cb0-b8d6-95ebbfd4fbaa" />
(A) Overview of the proposed DV-Prom framework: (a) DNABERT, (b) DNABERT-2, (c) DNABERT feature processing, (d) DNABERT-2 feature processing, and (e) Dual-View DNABERTs feature fusion. (B) Framework of the cross-validation ensemble strategy.

## Environment Installation

It is recommended to use Python 3.10 and install the corresponding PyTorch according to your CUDA version:
```bash
pip install -r requirements.txt
```
## Starting with Raw Data
```text
scripts/
  train.py # Main training script
  prepare_ipromp_datasets.py # Raw data -> .data
  precompute_tokens.py # .data -> .pt token cache
```
If you already have `Datasets_dfm` and `cache_tokens_dfm` ready (available at https://doi.org/10.5281/zenodo.22083569), you only need the main script for training; the other two scripts are used to regenerate the input cache from the raw data.
```bash
python model_dfm/prepare_ipromp_datasets.py --ipromp_root "./Benchmark Dataset" --out_root ./Datasets_dfm
python model_dfm/precompute_tokens.py  --datasets_root ./Datasets_dfm --cache_root ./cache_tokens_dfm
```
## DNABERTs Preparation
Please go to HuggingFace to access and download the pretrained DNABERT and DNABERT-2 models: https://huggingface.co/zhihan1996.
```text
DNA_bert_6/ # DNABERT 6-mer 
DNABERT-2-117M/ # DNABERT-2 
Datasets_dfm/ # train.data/test.data for 23 species
cache_tokens_dfm/ # train.pt/test.pt for each species
```
Please place the externally downloaded model directories in the project root directory and name them `DNA_bert_6` and
`DNABERT-2-117M`, or specify the paths using `--kmer_dir` and `--d2_dir`.

## Training and Testing
By default, train 23 species, perform 5-fold CV for each species, and then perform 5-fold ensemble on the independent test set:
```bash
python train.py -e 100 --es_patience 5 --batch_size 128 --lr 1e-4 --amp bf16
```
## Acknowledgements
We sincerely thank the authors of iPro-MP, Prompt, PromoterLCNN, and iPro-WAEL for open-source resources. We also thank the authors of DNABERT and DNABERT2, who provided powerful pre-trained models for downstream tasks.
