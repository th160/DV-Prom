import os
import sys
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, f1_score, roc_curve, auc,
                             matthews_corrcoef, confusion_matrix,
                             precision_recall_curve)
from transformers import BertModel, AutoModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_IPROMP_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
KMER_BERT_PATH = os.path.join(_IPROMP_ROOT, 'DNA_bert_6')
DNABERT2_PATH  = os.path.join(_IPROMP_ROOT, 'DNABERT-2-117M')

# List of species names
# name_list = ["Acinetobacter baumannii ATCC 17978", 
#              "Bradyrhizobium japonicum USDA 110", 
#              "Burkholderia cenocepacia J2315", 
#              "Campylobacter jejuni RM1221", 
#              "Campylobacter jejuni subsp. jejuni 81116", 
#              "Campylobacter jejuni subsp. jejuni 81-176", 
#              "Campylobacter jejuni subsp. jejuni NCTC 11168", 
#              "Corynebacterium diphtheriae NCTC 13129", 
#              "Corynebacterium glutamicum ATCC 13032", 
#              "Escherichia coli str K-12 substr. MG1655", 
#              "Haloferax volcanii DS2", 
#              "Helicobacter pylori strain 26695", 
#              "Nostoc sp. PCC7120", 
#              "Paenibacillus riograndensis SBR5", 
#              "Pseudomonas putida KT2440", 
#              "Shigella flexneri 5a str. M90T", 
#              "Sinorhizobium meliloti 1021", 
#              "Staphylococcus aureus subsp. aureus MW2", 
#              "Staphylococcus epidermidis ATCC 12228", 
#              "Synechococcus elongatus PCC 7942", 
#              "Thermococcus kodakarensis KOD1", 
#              "Xanthomonas campestris pv. campestrie B100", 
#              "Bacillus subtilis subsp. subtilis str. 168"]

name_list = ["A. baumannii ATCC 17978", 
             "B. japonicum USDA 110", 
             "B. cenocepacia J2315", 
             "C. jejuni RM1221", 
             "C. jejuni subsp. jejuni 81116", 
             "C. jejuni subsp. jejuni 81-176", 
             "C. jejuni subsp. jejuni NCTC 11168", 
             "C. diphtheriae NCTC 13129", 
             "C. glutamicum ATCC 13032", 
             "E. coli str K-12 substr. MG1655", 
             "H. volcanii DS2", 
             "H. pylori strain 26695", 
             "N. sp. PCC7120", 
             "P. riograndensis SBR5", 
             "P. putida KT2440", 
             "S. flexneri 5a str. M90T", 
             "S. meliloti 1021", 
             "S. aureus subsp. aureus MW2", 
             "S. epidermidis ATCC 12228", 
             "S. elongatus PCC 7942", 
             "T. kodakarensis KOD1", 
             "X. campestris pv. campestrie B100", 
             "B. subtilis subsp. subtilis str. 168"]
SPECIES_GROUPS = [
    ('01-07', range(1, 8)),
    ('08-15', range(8, 16)),
    ('16-23', range(16, 24)),
]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compute_seven_metrics(y_true, y_pred, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'AUC': auc_score,
            'AUPRC': auprc, 'MCC': mcc, 'F1': f1}

def pretty_name(sid):
    if 1 <= sid <= len(name_list):
        return name_list[sid - 1]
    return f'species {sid}'


def plot_single_dataset_curves(y_true, y_prob, sid, ds_name, out_dir):
    label_name = pretty_name(sid)
    os.makedirs(out_dir, exist_ok=True)

    # --- ROC ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc_score:.4f}')
    ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{label_name}')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{sid}_test_roc.png'), dpi=150)
    plt.close(fig)

    # --- PR ---
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, lw=2, label=f'AUPRC = {auprc:.4f}')
    ax.set_xlim([0.0, 1.02]); ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'{label_name}')
    ax.legend(loc='lower left'); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{sid}_test_pr.png'), dpi=150)
    plt.close(fig)


def plot_grouped_curves(curves, experiments_root):
    by_sid = {c['sid']: c for c in curves}

    for tag, sid_range in SPECIES_GROUPS:
        members = [by_sid[s] for s in sid_range if s in by_sid]
        if not members:
            print(f'[plot] group {tag}: no curves in range, skipped')
            continue

        # --- ROC ---
        fig, ax = plt.subplots(figsize=(7, 6))
        for c in members:
            fpr, tpr, _ = roc_curve(c['y_true'], c['y_prob'])
            a = auc(fpr, tpr)
            label = f' {pretty_name(c["sid"])}  (AUC={a:.3f})'
            ax.plot(fpr, tpr, lw=1.6, label=label)
        ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right', fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(experiments_root, f'AllSpecies-ROC-{tag}.png')
        fig.savefig(out, dpi=300); plt.close(fig)
        print(f'[plot] wrote {out}')

        # --- PR ---
        fig, ax = plt.subplots(figsize=(7, 6))
        for c in members:
            precision, recall, _ = precision_recall_curve(c['y_true'], c['y_prob'])
            a = auc(recall, precision)
            label = f' {pretty_name(c["sid"])}  (AUPRC={a:.3f})'
            ax.plot(recall, precision, lw=1.6, label=label)
        ax.set_xlim([0.0, 1.02]); ax.set_ylim([0.0, 1.02])
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.legend(loc='lower left', fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(experiments_root, f'AllSpecies-PR-{tag}.png')
        fig.savefig(out, dpi=300); plt.close(fig)
        print(f'[plot] wrote {out}')

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class MultiModalLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, p_dropout=0.3):
        super().__init__()
        self.self_attn_q = nn.MultiheadAttention(embed_dim=d_model,
                                               num_heads=num_heads,
                                               dropout=p_dropout,
                                               batch_first=True)
        self.self_attn_kv = nn.MultiheadAttention(embed_dim=d_model,
                                               num_heads=num_heads,
                                               dropout=p_dropout,
                                               batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model,
                                                num_heads=num_heads,
                                                dropout=p_dropout,
                                                batch_first=True)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.q_embedding_norm = nn.LayerNorm(d_model)   
        self.kv_embedding_norm = nn.LayerNorm(d_model)   
        self.cross_attn_norm  = nn.LayerNorm(d_model)
        self.norm             = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, q_embedding, kv_embedding, key_padding_mask=None):
        attn_out_q, _ = self.self_attn_q(q_embedding, q_embedding, q_embedding)
        q_embedding = self.q_embedding_norm(q_embedding + self.dropout(attn_out_q))

        attn_out_kv, _ = self.self_attn_kv(kv_embedding, kv_embedding, kv_embedding)
        kv_embedding = self.kv_embedding_norm(kv_embedding + self.dropout(attn_out_kv))

        attn_out, _ = self.cross_attn(
            query=q_embedding,
            key=kv_embedding,
            value=kv_embedding,
            key_padding_mask=key_padding_mask,  
        )
        q_embedding = self.cross_attn_norm(q_embedding + self.dropout(attn_out))

        ff_out = self.feed_forward(q_embedding)
        q_embedding = self.norm(q_embedding + self.dropout(ff_out))
        return q_embedding


class PoolingLayer(nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(d_model, d_model)
    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        x = self.dropout(x)
        x = torch.relu(x)
        return x


class CrossAttnClassifier(nn.Module):
    def __init__(self,
                 kmer_bert_dir=KMER_BERT_PATH,
                 dnabert2_dir=DNABERT2_PATH,
                 d_model=768,
                 num_heads=8,
                 d_ff=2048,
                 p_dropout=0.3):
        super().__init__()

        self.kmer_bert = BertModel.from_pretrained(kmer_bert_dir)
        self.dnabert2 = AutoModel.from_pretrained(dnabert2_dir,
                                                  trust_remote_code=True)

        assert self.kmer_bert.config.hidden_size == d_model, (
            f'kmer BERT hidden={self.kmer_bert.config.hidden_size} '
            f'!= d_model={d_model}')

        self.multi_modal_layer = MultiModalLayer(
            d_model=d_model, num_heads=num_heads,
            d_ff=d_ff, p_dropout=p_dropout,
        )
        self.pooling_layer = PoolingLayer(d_model=d_model, dropout=p_dropout)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, batch):
        kmer_out = self.kmer_bert(input_ids=batch['kmer_input_ids'],
                                  attention_mask=batch['kmer_attn_mask'],
                                  token_type_ids=batch['kmer_tok_type'])
        kv_embedding = kmer_out.last_hidden_state         
        d2_out = self.dnabert2(batch['dnabert2_input_ids'])[0]
        q_embedding = d2_out                              
        key_padding_mask = ~batch['kmer_attn_mask'].bool()   
        multi_modal_out = self.multi_modal_layer(
            q_embedding, kv_embedding, key_padding_mask=key_padding_mask)

        pooled = self.pooling_layer(multi_modal_out)      
        logits = self.classifier(pooled)                  
        return logits


class AttnDataset(Dataset):
    def __init__(self, pack, indices=None):
        self.p = pack
        self.idx = indices if indices is not None else list(range(len(pack['labels'])))
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        j = int(self.idx[i])
        return {
            'kmer_input_ids'    : self.p['kmer_input_ids'][j],
            'kmer_attn_mask'    : self.p['kmer_attn_mask'][j],
            'kmer_tok_type'     : self.p['kmer_tok_type'][j],
            'dnabert2_input_ids': self.p['dnabert2_input_ids'][j],
            'label'             : self.p['labels'][j],
        }


def attn_collate(batch):
    return (
        {
            'kmer_input_ids'    : torch.stack([b['kmer_input_ids']     for b in batch]),
            'kmer_attn_mask'    : torch.stack([b['kmer_attn_mask']     for b in batch]),
            'kmer_tok_type'     : torch.stack([b['kmer_tok_type']      for b in batch]),
            'dnabert2_input_ids': torch.stack([b['dnabert2_input_ids'] for b in batch]),
        },
        torch.stack([b['label'] for b in batch]),
    )


def build_model(args, device):
    net = CrossAttnClassifier(
        kmer_bert_dir=args.kmer_dir,
        dnabert2_dir=args.d2_dir,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        p_dropout=args.p_dropout,
    ).to(device)
    return net

@torch.no_grad()
def evaluate_loader(model, loader, device, amp_dtype=None):
    model.eval()
    use_amp = (amp_dtype is not None) and (device.type == 'cuda')
    ys, preds, probs = [], [], []
    for inputs, labels in loader:
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        if use_amp:
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                logits = model(inputs)
        else:
            logits = model(inputs)
        logits = logits.float()
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y = torch.argmax(logits, dim=1).cpu().numpy()
        ys.extend(labels.numpy().tolist())
        preds.extend(y.tolist())
        probs.extend(p.tolist())
    return (compute_seven_metrics(ys, preds, probs),
            np.array(probs), np.array(ys))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        return torch.mean(self.alpha * (1 - pt) ** self.gamma * BCE_loss)

def train_one_fold(model, train_loader, val_loader, epochs, lr,
                   save_path, device, log_prefix='',
                   es_patience=5, es_min_delta=1e-4, amp_dtype=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
    use_amp = (amp_dtype is not None) and (device.type == 'cuda')
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    best_metrics, best_auc, best_epoch = None, 0.0, -1
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    logits = model(inputs)
                    loss = loss_fn(logits, labels)
                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                logits = model(inputs)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))

        metrics, _, _ = evaluate_loader(model, val_loader, device, amp_dtype)
        print(f'{log_prefix}Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} '
              f'| val Sn:{metrics["Sn"]:.4f} Sp:{metrics["Sp"]:.4f} '
              f'ACC:{metrics["ACC"]:.4f} AUC:{metrics["AUC"]:.4f} '
              f'AUPRC:{metrics["AUPRC"]:.4f} MCC:{metrics["MCC"]:.4f} '
              f'F1:{metrics["F1"]:.4f}')

        if metrics['AUC'] > best_auc + es_min_delta:
            best_metrics = metrics
            best_auc = metrics['AUC']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= es_patience:
                print(f'{log_prefix}  early stop @ epoch {epoch+1} '
                      f'(best AUC {best_auc:.4f} @ epoch {best_epoch})')
                break
    return best_metrics, best_auc, best_epoch

def run_one_species(species_id, ds_name, args, device):
    cache_dir = os.path.join(args.cache_root, ds_name)
    train_pt = os.path.join(cache_dir, 'train.pt')
    test_pt  = os.path.join(cache_dir, 'test.pt')
    if not (os.path.exists(train_pt) and os.path.exists(test_pt)):
        raise FileNotFoundError(f'Missing cached tokens for {ds_name}. '
                                'Run precompute_tokens.py first.')

    train_pack = torch.load(train_pt, weights_only=False)
    test_pack  = torch.load(test_pt,  weights_only=False)
    N_train = len(train_pack['labels'])
    print(f'[{ds_name}] N_train={N_train}, N_test={len(test_pack["labels"])}')

    exp_dir = os.path.join(args.experiments_root, ds_name)
    model_dir = os.path.join(exp_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)

    fold_results = []
    fold_ckpts = []
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(range(N_train)), start=1):
        print(f'\n=== [{ds_name}] fold {fold_idx}/{args.folds} '
              f'(train={len(tr_idx)}  val={len(va_idx)}) ===')

        tr_loader = DataLoader(AttnDataset(train_pack, indices=tr_idx.tolist()),
                               batch_size=args.batch_size, shuffle=True,
                               collate_fn=attn_collate,
                               num_workers=args.num_workers, pin_memory=True)
        va_loader = DataLoader(AttnDataset(train_pack, indices=va_idx.tolist()),
                               batch_size=args.batch_size, shuffle=False,
                               collate_fn=attn_collate,
                               num_workers=args.num_workers, pin_memory=True)

        save_path = os.path.join(model_dir, f'Species{species_id}_fold{fold_idx}.pth')
        net = build_model(args, device)

        t0 = time.time()
        best_metrics, best_auc, best_epoch = train_one_fold(
            net, tr_loader, va_loader,
            epochs=args.epochs, lr=args.lr,
            save_path=save_path, device=device,
            log_prefix=f'    [sid{species_id} f{fold_idx}] ',
            es_patience=args.es_patience,
            es_min_delta=args.es_min_delta,
            amp_dtype=args.amp_dtype,
        )
        dt = time.time() - t0
        best_metrics = {**best_metrics, 'Fold': fold_idx,
                        'BestEpoch': best_epoch, 'TrainSec': round(dt, 2)}
        fold_results.append(best_metrics)
        fold_ckpts.append(save_path)
        print(f'[{ds_name}] fold{fold_idx} best AUC={best_auc:.4f} '
              f'@epoch {best_epoch} in {dt:.1f}s')

        del net
        torch.cuda.empty_cache()

    cv_df = pd.DataFrame(fold_results)
    cv_csv = os.path.join(exp_dir, f'{species_id}_5fold.csv')
    cv_df.to_csv(cv_csv, index=False)
    print(f'\n[{ds_name}] 5-fold CV saved -> {cv_csv}')
    print(cv_df[['Fold', 'Sn', 'Sp', 'ACC', 'AUC', 'AUPRC', 'MCC', 'F1', 'TrainSec']]
          .to_string(index=False))


    test_loader = DataLoader(AttnDataset(test_pack),
                             batch_size=args.batch_size, shuffle=False,
                             collate_fn=attn_collate,
                             num_workers=args.num_workers, pin_memory=True)
    true_labels = test_pack['labels'].numpy()
    aggregated = np.zeros(len(test_pack['labels']), dtype=np.float64)

    t1 = time.time()
    for ck_path in fold_ckpts:
        net = build_model(args, device)
        net.load_state_dict(torch.load(ck_path, map_location=device))
        _, probs, _ = evaluate_loader(net, test_loader, device, args.amp_dtype)
        aggregated += probs
        del net
        torch.cuda.empty_cache()

    aggregated /= len(fold_ckpts)
    preds = (aggregated >= 0.5).astype(int)
    dt1 = time.time() - t1
    print(f'Independent test time: {dt1:.1f}s')
    test_metrics = compute_seven_metrics(true_labels.tolist(),
                                         preds.tolist(),
                                         aggregated.tolist())
    test_metrics['SpeciesID'] = species_id
    test_metrics['Dataset']   = ds_name
    test_metrics['TestSec'] = dt1
    test_df = pd.DataFrame([test_metrics])
    test_csv = os.path.join(exp_dir, f'{species_id}_test.csv')
    test_df.to_csv(test_csv, index=False)
    print(f'[{ds_name}] independent test (5-fold ensemble) -> {test_csv}')
    print(test_df[['Sn', 'Sp', 'ACC', 'AUC', 'AUPRC', 'MCC', 'F1', 'TestSec']]
          .to_string(index=False))


    seqs = test_pack.get('sequences') if isinstance(test_pack, dict) else None
    if seqs is None or len(seqs) != len(true_labels):
        seqs = [''] * len(true_labels)
    pred_csv = os.path.join(exp_dir, f'{species_id}_test_predictions.csv')
    pd.DataFrame({
        'Index'      : np.arange(len(true_labels)),
        'Sequence'   : seqs,
        'Label'      : true_labels.astype(int),
        'Probability': aggregated,                          
        'Prediction' : preds.astype(int),                   
        'Correct'    : (preds == true_labels).astype(int),  
    }).to_csv(pred_csv, index=False)
    print(f'[{ds_name}] per-sample predictions -> {pred_csv}  '
          f'(N={len(true_labels)}, correct={int((preds == true_labels).sum())})')

    
    plot_single_dataset_curves(true_labels, aggregated,
                               sid=species_id, ds_name=ds_name, out_dir=exp_dir)
    print(f'[{ds_name}] ROC/PR PNGs -> {exp_dir}\\{species_id}_test_roc.png, '
          f'{species_id}_test_pr.png')

    
    return cv_df, test_metrics, aggregated, true_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', '--epochs', type=int, default=100)   
    ap.add_argument('--batch_size', type=int, default=32,
                    help='Two BERTs + a cross-attn layer -> keep it moderate.')
    ap.add_argument('--lr', type=float, default=1e-5)          
    ap.add_argument('--folds', type=int, default=5)            

    # early-stopping (paper Method: patience=5, min_delta=1e-4 on val AUC)
    ap.add_argument('--es_patience', type=int, default=5)
    ap.add_argument('--es_min_delta', type=float, default=1e-4)

    ap.add_argument('--amp', choices=['off', 'fp16', 'bf16'], default='bf16')
    ap.add_argument('--fast_cudnn', action='store_true')
    ap.add_argument('--num_workers', type=int, default=0)

    ap.add_argument('--d_model', type=int, default=768)
    ap.add_argument('--num_heads', type=int, default=8)
    ap.add_argument('--d_ff', type=int, default=2048)
    ap.add_argument('--p_dropout', type=float, default=0.3)

    ap.add_argument('--kmer_dir', type=str, default=KMER_BERT_PATH)
    ap.add_argument('--d2_dir',   type=str, default=DNABERT2_PATH)
    ap.add_argument('--datasets_root', type=str,
                    default=os.path.join(_IPROMP_ROOT, 'Datasets_dfm'))
    ap.add_argument('--cache_root', type=str,
                    default=os.path.join(_IPROMP_ROOT, 'cache_tokens_dfm'))
    ap.add_argument('--experiments_root', type=str,
                    default=os.path.join(_IPROMP_ROOT,
                                         'models_dir'))
    ap.add_argument('--single_dataset', type=str, default=None)
    ap.add_argument('--start_species', type=int, default=1)
    ap.add_argument('--end_species',   type=int, default=23)
    args = ap.parse_args()

    args.amp_dtype = {'off': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.amp]

    set_seed(42)
    if args.fast_cudnn:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    os.makedirs(args.experiments_root, exist_ok=True)

    with open(os.path.join(args.datasets_root, 'FILESLIST.txt')) as f:
        all_ds = [line.strip() for line in f if line.strip()]

    def sid_of(name):
        return int(name.split('_')[1])

    if args.single_dataset:
        selected = [d for d in all_ds if d == args.single_dataset]
        if not selected:
            raise ValueError(f'{args.single_dataset} not in FILESLIST.txt')
    else:
        selected = [d for d in all_ds
                    if args.start_species <= sid_of(d) <= args.end_species]

    all_cv_rows, all_test_rows = [], []
    all_curves = []                       
    for ds_name in selected:
        sid = sid_of(ds_name)
        print('\n' + '=' * 80)
        print(f'** Processing species {sid}: {ds_name} **')
        print('=' * 80)
        try:
            cv_df, test_row, agg_probs, true_lbls = run_one_species(
                sid, ds_name, args, device)
            cv_mean = cv_df[['Sn', 'Sp', 'ACC', 'AUC', 'AUPRC', 'MCC', 'F1']].mean().to_dict()
            cv_mean.update({'SpeciesID': sid, 'Dataset': ds_name})
            all_cv_rows.append(cv_mean)
            all_test_rows.append(test_row)
            all_curves.append({
                'sid': sid, 'ds_name': ds_name,
                'y_true': true_lbls, 'y_prob': agg_probs,
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'[!!] Species {sid} failed: {e}')

    if all_cv_rows:
        pd.DataFrame(all_cv_rows).to_csv(
            os.path.join(args.experiments_root, 'AllSpecies-5fold.csv'), index=False)
    if all_test_rows:
        pd.DataFrame(all_test_rows).to_csv(
            os.path.join(args.experiments_root, 'AllSpecies-test.csv'), index=False)


    if all_curves:
        plot_grouped_curves(all_curves, args.experiments_root)

    print('\nDone.')


if __name__ == '__main__':
    main()
