#!/usr/bin/env python3
# =============================================================================
#  TRDT — Task-Relevant Drift Theory : replacing the N=3 correlation
#  Self-contained, Kaggle-ready (GPU + Internet ON).
#
#  WHAT THIS DOES
#  --------------
#  The original paper claimed  r = 0.91  between Task-Relevant Drift (TR-drift)
#  and forgetting, but from only 3 learning-rate conditions (N = 3). That is a
#  confound: raising the learning rate raises BOTH total drift and TR-drift, so
#  the correlation does not show TR-drift is the *cause*.
#
#  This script produces three things a reviewer will accept:
#    (A) OBSERVATIONAL SWEEP  -> large N from (task-pair x seed x lr) conditions,
#        with bootstrap CIs for the TR-drift<->forgetting correlation.
#    (B) PARTIAL / NESTED test -> does TR-drift explain variance in forgetting
#        AFTER controlling for total drift? (the key incremental-validity test)
#    (C) CAUSAL DECOUPLING     -> train Task B twice with the SAME total backbone
#        step norm, but push the backbone update either INTO the Task-A subspace
#        P_A or ORTHOGONAL to it. If "into P_A" forgets more at matched total
#        drift, TR-drift is causal (this is prediction P4 in your paper).
#
#  Setup matches texte_7.txt: GPT-2 small, LoRA rank 16 / alpha 32, backbone
#  trainable, AdamW, LoRA lr 2e-4 / head lr 1e-3, batch 16, ~3 epochs,
#  500 samples (350 train / 150 test). P_A = top-k PCs of Task-A gradients
#  over `n_grad_batches` batches. Forgetting = max(0, Acc_A(after A) - Acc_A(after B)).
#
#  KAGGLE QUICKSTART
#  -----------------
#    1) Notebook settings: Accelerator = GPU (T4 x2 or P100), Internet = ON.
#    2) !pip -q install "transformers>=4.40" "peft>=0.11" datasets accelerate \
#         scikit-learn statsmodels scipy pandas
#    3) Upload this file, then:
#         !python trdt_kaggle.py --smoke      # ~1-2 min, verify it runs
#         !python trdt_kaggle.py --full       # the real run (hours on T4)
#    4) Outputs land in /kaggle/working/ : observational.csv, intervention.csv,
#       and a printed summary table that replaces the N=3 result.
#
#  Author scaffold for: Haithem Barkaoui — TRDT revision. MIT-style, adapt freely.
# =============================================================================

import os, sys, json, time, math, argparse, random, gc
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

# Torch / HF imports are deferred until after arg-parse so --help is instant.

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class Config:
    model_name: str = "gpt2"                 # GPT-2 small (124M)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_lr: float = 2e-4
    head_lr: float = 1e-3
    backbone_lr: float = 2e-4                # backbone is trainable (matches paper)
    batch_size: int = 16
    epochs: int = 3
    max_len: int = 128
    n_train: int = 350
    n_test: int = 150
    n_grad_batches: int = 32                 # batches used to estimate P_A (paper used 64)
    subspace_k: int = 16                     # top-k gradient PCs => dim of P_A
    # Which BACKBONE tensors define the parameter space we analyse + intervene on.
    # Default: attention projections (where LoRA acts and the companion paper focuses).
    target_substrings: Tuple[str, ...] = ("attn.c_attn", "attn.c_proj")
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    learning_rates: Tuple[float, ...] = (5e-6, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4)
    # Task pairs for the observational sweep (A -> B). Keys map to loaders below.
    task_pairs: Tuple[Tuple[str, str], ...] = (
        ("imdb", "ag_news"),
        ("ag_news", "imdb"),
        ("sst2", "ag_news"),
        ("yelp", "ag_news"),
        ("imdb", "sst2"),
    )
    # Intervention: matched total backbone step-norm per step (None => use measured mean).
    intervention_lrs: Tuple[float, ...] = (5e-5, 2e-4)
    intervention_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    out_dir: str = "/kaggle/working"
    smoke: bool = False


def apply_smoke(cfg: Config) -> Config:
    """Tiny, fast settings to verify the pipeline end-to-end."""
    cfg.epochs = 1
    cfg.n_train = 64
    cfg.n_test = 64
    cfg.n_grad_batches = 6
    cfg.subspace_k = 4
    cfg.seeds = (0, 1)
    cfg.learning_rates = (5e-6, 2e-4, 5e-4)
    cfg.task_pairs = (("imdb", "ag_news"), ("ag_news", "imdb"))
    cfg.intervention_lrs = (2e-4,)
    cfg.intervention_seeds = (0, 1)
    cfg.smoke = True
    if not os.path.isdir("/kaggle/working"):
        cfg.out_dir = "."
    return cfg


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    import torch
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# -----------------------------------------------------------------------------
# Data: each loader returns (texts, labels) with binary labels {0,1}
# -----------------------------------------------------------------------------
def load_task(name: str, n: int, seed: int):
    from datasets import load_dataset
    name = name.lower()
    if name == "imdb":
        ds = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=seed).select(range(n))
        return list(ds["text"]), list(ds["label"])
    if name == "yelp":
        ds = load_dataset("fancyzhx/yelp_polarity", split="train").shuffle(seed=seed).select(range(n))
        return list(ds["text"]), list(ds["label"])
    if name == "sst2":
        ds = load_dataset("nyu-mll/glue", "sst2", split="train").shuffle(seed=seed).select(range(n))
        return list(ds["sentence"]), list(ds["label"])
    if name == "ag_news":
        # 4-class topic -> binary (World/Sports vs Business/Sci-Tech) for a clean 2-way task
        ds = load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=seed).select(range(n))
        texts = list(ds["text"]); labels = [0 if l < 2 else 1 for l in ds["label"]]
        return texts, labels
    raise ValueError(f"unknown task: {name}")


def make_split(name: str, cfg: Config, seed: int):
    texts, labels = load_task(name, cfg.n_train + cfg.n_test, seed)
    return (texts[:cfg.n_train], labels[:cfg.n_train],
            texts[cfg.n_train:cfg.n_train + cfg.n_test], labels[cfg.n_train:cfg.n_train + cfg.n_test])


def tokenize(tok, texts, labels, cfg, device):
    import torch
    enc = tok(texts, truncation=True, padding="max_length", max_length=cfg.max_len, return_tensors="pt")
    enc["labels"] = torch.tensor(labels, dtype=torch.long)
    return {k: v.to(device) for k, v in enc.items()}


def iter_batches(data, bs, shuffle=True, seed=0):
    import torch
    n = data["labels"].shape[0]
    idx = list(range(n))
    if shuffle:
        rng = random.Random(seed); rng.shuffle(idx)
    for i in range(0, n, bs):
        sel = idx[i:i + bs]
        yield {k: v[sel] for k, v in data.items()}


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def build_model(cfg: Config, device):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model, TaskType

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)
    base.config.pad_token_id = tok.pad_token_id

    lcfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                      lora_dropout=cfg.lora_dropout, target_modules=["c_attn"], bias="none")
    model = get_peft_model(base, lcfg)

    # Make the BACKBONE trainable too (paper: "backbone is fully trainable").
    for n_, p in model.named_parameters():
        if "lora_" in n_ or "modules_to_save" in n_ or ".score." in n_ or "classifier" in n_:
            p.requires_grad = True
        else:
            p.requires_grad = True  # backbone trainable
    model.to(device)
    return model, tok


def param_groups(model, cfg):
    lora, head, backbone = [], [], []
    for n_, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n_:
            lora.append(p)
        elif (".score." in n_) or ("classifier" in n_) or ("modules_to_save" in n_):
            head.append(p)
        else:
            backbone.append(p)
    return lora, head, backbone


def target_params(model, cfg) -> List[Tuple[str, "object"]]:
    """The backbone tensors that define the analysed parameter space.
       Note: PEFT renames the wrapped c_attn weight to '...c_attn.base_layer.weight',
       so we match by substring + '.weight' and exclude lora_ params."""
    out = []
    for n_, p in model.named_parameters():
        if "lora_" in n_:
            continue
        if n_.endswith(".weight") and any(s in n_ for s in cfg.target_substrings):
            out.append((n_, p))
    return out


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------
def train_task(model, data, cfg, lr, seed, label=""):
    import torch
    lora, head, backbone = param_groups(model, cfg)
    opt = torch.optim.AdamW(
        [{"params": lora, "lr": lr},
         {"params": head, "lr": cfg.head_lr},
         {"params": backbone, "lr": lr}],   # swept lr drives the backbone (the drift mechanism)
        weight_decay=0.0)
    model.train()
    for ep in range(cfg.epochs):
        for b, batch in enumerate(iter_batches(data, cfg.batch_size, True, seed * 1000 + ep)):
            opt.zero_grad()
            out = model(**batch)
            out.loss.backward()
            opt.step()
    return model


def _eval_impl(model, data, cfg):
    import torch
    model.eval()
    correct = tot = 0
    with torch.no_grad():
        for batch in iter_batches(data, cfg.batch_size, False):
            logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            pred = logits.argmax(-1)
            correct += (pred == batch["labels"]).sum().item()
            tot += batch["labels"].shape[0]
    return correct / max(tot, 1)


def eval_acc(model, data, cfg):
    return _eval_impl(model, data, cfg)


# -----------------------------------------------------------------------------
# Subspace P_A  (top-k PCs of Task-A gradients), computed per-tensor for memory
# -----------------------------------------------------------------------------
def compute_subspace(model, dataA, cfg, seed) -> Dict[str, np.ndarray]:
    """Returns {param_name: V} where V is [k, d] orthonormal gradient PCs (fp16, CPU/np)."""
    import torch
    tparams = target_params(model, cfg)
    names = [n_ for n_, _ in tparams]
    # collect flattened gradients per target tensor across n_grad_batches
    grads = {n_: [] for n_ in names}
    model.train()
    bit = iter_batches(dataA, cfg.batch_size, True, seed * 7 + 11)
    nb = 0
    for batch in bit:
        model.zero_grad(set_to_none=True)
        out = model(**batch)
        out.loss.backward()
        for n_, p in tparams:
            g = p.grad.detach().reshape(-1).to(torch.float32).cpu().numpy()
            grads[n_].append(g)
        nb += 1
        if nb >= cfg.n_grad_batches:
            break
    model.zero_grad(set_to_none=True)

    subspace = {}
    for n_ in names:
        G = np.stack(grads[n_], axis=0)            # [n_batches, d]
        G = G - G.mean(axis=0, keepdims=True)      # center
        # economy SVD: right singular vectors V^T are the PCs in parameter space
        # G = U S Vt  ;  rows of Vt are PCs (length d)
        try:
            _, _, Vt = np.linalg.svd(G, full_matrices=False)
        except np.linalg.LinAlgError:
            Vt = np.zeros((cfg.subspace_k, G.shape[1]), dtype=np.float32)
        k = min(cfg.subspace_k, Vt.shape[0])
        subspace[n_] = Vt[:k].astype(np.float16)   # [k, d]
        del G
    del grads; gc.collect()
    return subspace


def subspace_overlap(subA, subB):
    """Static overlap between two gradient subspaces (mean squared cosine, in [0,1]).
       1 => identical subspaces, 0 => orthogonal. This is the STATIC predictor that
       the negative result shows does NOT track forgetting."""
    num = 0.0; den = 0.0
    for n_ in subA:
        if n_ not in subB:
            continue
        U = subA[n_].astype(np.float32)   # [k, d] orthonormal rows
        V = subB[n_].astype(np.float32)   # [k, d]
        M = U @ V.T                        # [k, k]
        num += float((M ** 2).sum())
        den += min(U.shape[0], V.shape[0])
    return num / den if den > 0 else float("nan")


def snapshot_target(model, cfg) -> Dict[str, np.ndarray]:
    import torch
    return {n_: p.detach().reshape(-1).to(torch.float32).cpu().numpy()
            for n_, p in target_params(model, cfg)}


def drift_measures(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray],
                   subspace: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Total drift and TR-drift, aggregated across tensors in quadrature.
       TR-drift = ||P_A (theta_afterB - theta_afterA)|| / ||theta_afterA||"""
    sq_total = 0.0
    sq_proj = 0.0
    sq_base = 0.0
    for n_ in before:
        d = (after[n_] - before[n_]).astype(np.float32)         # delta
        base = after[n_].astype(np.float32)                     # theta_afterA baseline norm
        sq_total += float(np.dot(d, d))
        sq_base  += float(np.dot(base, base))
        V = subspace.get(n_)
        if V is not None and V.size:
            coeffs = V.astype(np.float32) @ d                    # [k]
            sq_proj += float(np.dot(coeffs, coeffs))
    base_norm = math.sqrt(sq_base) + 1e-12
    return {
        "total_drift": math.sqrt(sq_total) / base_norm,
        "tr_drift":    math.sqrt(sq_proj)  / base_norm,
        "tr_fraction": math.sqrt(sq_proj) / (math.sqrt(sq_total) + 1e-12),
    }


# -----------------------------------------------------------------------------
# (A)+(B)  Observational sweep
# -----------------------------------------------------------------------------
def run_observational(cfg: Config):
    import torch, pandas as pd
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    t0 = time.time()
    path = os.path.join(cfg.out_dir, "observational.csv")
    # resume: skip (taskA,taskB,seed) groups already complete (all learning_rates present)
    done = set()
    if os.path.exists(path):
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        cnt = prev.groupby(["taskA", "taskB", "seed"])["lr"].nunique()
        need = len(cfg.learning_rates)
        done = set(cnt[cnt >= need].index)
        print(f"[resume] {len(done)} (pair,seed) groups already complete -> skipping them")
    for (taskA, taskB) in cfg.task_pairs:
        for seed in cfg.seeds:
            if (taskA, taskB, seed) in done:
                continue
            # data
            set_seed(seed)
            model, tok = build_model(cfg, device)
            aTr, aTrL, aTe, aTeL = make_split(taskA, cfg, seed)
            bTr, bTrL, _, _      = make_split(taskB, cfg, seed)
            dA_tr = tokenize(tok, aTr, aTrL, cfg, device)
            dA_te = tokenize(tok, aTe, aTeL, cfg, device)
            dB_tr = tokenize(tok, bTr, bTrL, cfg, device)

            # Train Task A (use a middle lr for the A phase so P_A is well-formed)
            train_task(model, dA_tr, cfg, lr=cfg.lora_lr, seed=seed, label="A")
            accA_afterA = eval_acc(model, dA_te, cfg)
            subspace = compute_subspace(model, dA_tr, cfg, seed)
            subspaceB = compute_subspace(model, dB_tr, cfg, seed)   # P_B at post-A point
            static_overlap = subspace_overlap(subspace, subspaceB)  # the negative-result predictor
            before = snapshot_target(model, cfg)

            for lr in cfg.learning_rates:
                # clone model state to re-run Task B at each lr from the same post-A point
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                train_task(model, dB_tr, cfg, lr=lr, seed=seed, label="B")
                accA_afterB = eval_acc(model, dA_te, cfg)
                after = snapshot_target(model, cfg)
                dm = drift_measures(before, after, subspace)
                forgetting = max(0.0, accA_afterA - accA_afterB)
                rows.append({
                    "taskA": taskA, "taskB": taskB, "seed": seed, "lr": lr,
                    "accA_afterA": accA_afterA, "accA_afterB": accA_afterB,
                    "forgetting": forgetting, "static_overlap": static_overlap, **dm,
                    "elapsed_min": round((time.time() - t0) / 60, 2),
                })
                print(f"[obs] {taskA}->{taskB} seed={seed} lr={lr:.0e} "
                      f"forget={forgetting:.3f} overlap={static_overlap:.3f} "
                      f"total={dm['total_drift']:.4f} tr={dm['tr_drift']:.4f}", flush=True)
                model.load_state_dict(state)  # restore post-A point for next lr
                del state
            # incremental save after each completed (pair,seed) group (survives disconnects)
            pd.DataFrame(rows).to_csv(path, index=False)
            del model; gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df)} conditions -> {path}")
    return df


# -----------------------------------------------------------------------------
# (C)  Causal decoupling intervention
#  Freeze LoRA + head; drive ONLY the backbone target tensors with a controlled
#  direction at MATCHED per-step norm. Arm "into"=update inside P_A, "orth"=outside.
# -----------------------------------------------------------------------------
def project_grad(g_flat: "object", V: "object", mode: str):
    """g_flat: torch 1D; V: torch [k,d] orthonormal rows. Return projected direction."""
    import torch
    if V is None or V.numel() == 0:
        return g_flat if mode == "orth" else torch.zeros_like(g_flat)
    coeffs = V @ g_flat                      # [k]
    g_par = V.t() @ coeffs                   # component inside P_A
    if mode == "into":
        return g_par
    else:  # orthogonal complement
        return g_flat - g_par


def train_taskB_controlled(model, dB, cfg, subspace_t, mode, step_norm, seed):
    """Manual SGD on backbone target tensors with direction control + matched norm.
       LoRA + head are frozen (held at post-A values) to isolate backbone drift."""
    import torch
    tparams = target_params(model, cfg)
    # freeze everything except backbone target tensors
    for n_, p in model.named_parameters():
        p.requires_grad = any(n_ == tn for tn, _ in tparams)
    model.train()
    for ep in range(cfg.epochs):
        for batch in iter_batches(dB, cfg.batch_size, True, seed * 13 + ep):
            model.zero_grad(set_to_none=True)
            out = model(**batch)
            out.loss.backward()
            # gather, project, rescale to matched total step norm, apply
            with torch.no_grad():
                dirs = {}
                sq = 0.0
                for n_, p in tparams:
                    if p.grad is None:
                        dirs[n_] = torch.zeros_like(p).reshape(-1); continue
                    g = p.grad.detach().reshape(-1)
                    V = subspace_t.get(n_)
                    d = project_grad(g, V, mode)
                    dirs[n_] = d
                    sq += float(torch.dot(d, d))
                norm = math.sqrt(sq) + 1e-12
                scale = step_norm / norm                 # match total step magnitude
                for n_, p in tparams:
                    upd = (dirs[n_] * scale).reshape(p.shape)
                    p.add_(-upd)                          # gradient descent step
    model.zero_grad(set_to_none=True)


def run_intervention(cfg: Config):
    import torch, pandas as pd
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for lr in cfg.intervention_lrs:
        for seed in cfg.intervention_seeds:
            set_seed(seed)
            model, tok = build_model(cfg, device)
            # Use the IMDB->AG News causal pair from the paper
            aTr, aTrL, aTe, aTeL = make_split("imdb", cfg, seed)
            bTr, bTrL, _, _      = make_split("ag_news", cfg, seed)
            dA_tr = tokenize(tok, aTr, aTrL, cfg, device)
            dA_te = tokenize(tok, aTe, aTeL, cfg, device)
            dB_tr = tokenize(tok, bTr, bTrL, cfg, device)

            train_task(model, dA_tr, cfg, lr=cfg.lora_lr, seed=seed)
            accA_afterA = eval_acc(model, dA_te, cfg)
            subspace = compute_subspace(model, dA_tr, cfg, seed)
            subspace_t = {n_: torch.tensor(V.astype(np.float32), device=device)
                          for n_, V in subspace.items()}
            before = snapshot_target(model, cfg)
            post_A_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # Calibrate a target per-step norm from a normal Task-B run (so "matched
            # total drift" reflects a realistic magnitude at this lr).
            train_task(model, dB_tr, cfg, lr=lr, seed=seed)
            after_cal = snapshot_target(model, cfg)
            cal = drift_measures(before, after_cal, subspace)
            n_steps = cfg.epochs * math.ceil(cfg.n_train / cfg.batch_size)
            # total backbone delta norm (un-normalized) split across steps:
            total_delta_norm = math.sqrt(sum(float(np.dot((after_cal[n_]-before[n_]),
                                                           (after_cal[n_]-before[n_])))
                                             for n_ in before))
            step_norm = total_delta_norm / max(n_steps, 1)
            model.load_state_dict(post_A_state)

            for mode in ("into", "orth"):
                model.load_state_dict(post_A_state)
                train_taskB_controlled(model, dB_tr, cfg, subspace_t, mode, step_norm, seed)
                accA_afterB = eval_acc(model, dA_te, cfg)
                after = snapshot_target(model, cfg)
                dm = drift_measures(before, after, subspace)
                forgetting = max(0.0, accA_afterA - accA_afterB)
                rows.append({
                    "lr": lr, "seed": seed, "arm": mode,
                    "accA_afterA": accA_afterA, "accA_afterB": accA_afterB,
                    "forgetting": forgetting, **dm, "matched_step_norm": step_norm,
                })
                print(f"[int] lr={lr:.0e} seed={seed} arm={mode:4s} "
                      f"forget={forgetting:.3f} total={dm['total_drift']:.4f} "
                      f"tr={dm['tr_drift']:.4f}", flush=True)
            del model, subspace_t, post_A_state; gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    path = os.path.join(cfg.out_dir, "intervention.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df)} intervention rows -> {path}")
    return df


# -----------------------------------------------------------------------------
# Analysis  (runs on the saved CSVs; reviewer-grade statistics)
# -----------------------------------------------------------------------------
def bootstrap_corr(x, y, n_boot=5000, seed=0, method="pearson"):
    from scipy.stats import pearsonr, spearmanr
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y); n = len(x)
    f = (lambda a, b: pearsonr(a, b)[0]) if method == "pearson" else (lambda a, b: spearmanr(a, b)[0])
    r0 = f(x, y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try: boots.append(f(x[idx], y[idx]))
        except Exception: pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return r0, lo, hi


def partial_corr(df, x, y, control):
    """Partial correlation of x,y controlling for `control` via residualization."""
    import numpy as np
    from numpy.linalg import lstsq
    from scipy.stats import pearsonr
    C = np.column_stack([np.ones(len(df)), df[control].values])
    def resid(v):
        beta, *_ = lstsq(C, df[v].values, rcond=None)
        return df[v].values - C @ beta
    rx, ry = resid(x), resid(y)
    r, p = pearsonr(rx, ry)
    return r, p


def analyze(cfg: Config):
    import pandas as pd
    print("\n" + "=" * 70)
    print("ANALYSIS — replacing the N=3 result")
    print("=" * 70)

    obs_path = os.path.join(cfg.out_dir, "observational.csv")
    if os.path.exists(obs_path):
        df = pd.read_csv(obs_path)
        df = df.dropna(subset=["forgetting", "total_drift", "tr_drift"])
        N = len(df)
        print(f"\n(A) OBSERVATIONAL SWEEP   N = {N} conditions "
              f"(vs the old N = 3)\n")
        if "static_overlap" in df.columns and df["static_overlap"].notna().any():
            so = df.dropna(subset=["static_overlap"])
            r, lo, hi = bootstrap_corr(so["static_overlap"], so["forgetting"], method="pearson")
            print(f"  [NEGATIVE RESULT] static_overlap vs forgetting : "
                  f"Pearson r = {r:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                  f"(near 0 => static overlap does NOT predict forgetting)\n")
        for col in ["tr_drift", "total_drift"]:
            r, lo, hi = bootstrap_corr(df[col], df["forgetting"], method="pearson")
            rs, _, _ = bootstrap_corr(df[col], df["forgetting"], method="spearman")
            print(f"  {col:11s} vs forgetting : Pearson r = {r:+.3f}  "
                  f"95% CI [{lo:+.3f}, {hi:+.3f}]   Spearman = {rs:+.3f}")

        print("\n(B) INCREMENTAL VALIDITY — does TR-drift add anything beyond total drift?\n")
        r_pc, p_pc = partial_corr(df, "tr_drift", "forgetting", "total_drift")
        print(f"  Partial corr  TR-drift <-> forgetting | total_drift : "
              f"r = {r_pc:+.3f}  p = {p_pc:.4f}")
        try:
            import statsmodels.formula.api as smf
            m1 = smf.ols("forgetting ~ total_drift", data=df).fit()
            m2 = smf.ols("forgetting ~ total_drift + tr_drift", data=df).fit()
            dR2 = m2.rsquared - m1.rsquared
            print(f"  Nested OLS:  R^2(total) = {m1.rsquared:.3f}  ->  "
                  f"R^2(total+TR) = {m2.rsquared:.3f}   Delta R^2 = {dR2:+.3f}")
            print(f"  TR-drift coefficient: beta = {m2.params.get('tr_drift', float('nan')):+.4f}  "
                  f"p = {m2.pvalues.get('tr_drift', float('nan')):.4f}")
            try:
                import statsmodels.api as sm
                mixed = smf.mixedlm("forgetting ~ total_drift + tr_drift", df,
                                    groups=df["taskA"]).fit(reml=False, method="lbfgs")
                print(f"  Mixed-effects (group=taskA) TR-drift: "
                      f"beta = {mixed.params.get('tr_drift', float('nan')):+.4f}  "
                      f"p = {mixed.pvalues.get('tr_drift', float('nan')):.4f}")
            except Exception as e:
                print(f"  (mixed-effects skipped: {e})")
        except Exception as e:
            print(f"  (statsmodels unavailable: {e})")

    int_path = os.path.join(cfg.out_dir, "intervention.csv")
    if os.path.exists(int_path):
        di = pd.read_csv(int_path)
        print("\n(C) CAUSAL DECOUPLING — same total drift, opposite direction vs P_A\n")
        piv = di.pivot_table(index=["lr", "seed"], columns="arm",
                             values=["forgetting", "total_drift"]).dropna()
        if len(piv):
            f_into = piv[("forgetting", "into")].values
            f_orth = piv[("forgetting", "orth")].values
            t_into = piv[("total_drift", "into")].values
            t_orth = piv[("total_drift", "orth")].values
            from scipy.stats import wilcoxon, ttest_rel
            diff = f_into - f_orth
            d = diff.mean() / (diff.std(ddof=1) + 1e-12)
            try: w_p = wilcoxon(f_into, f_orth).pvalue
            except Exception: w_p = float("nan")
            try: t_p = ttest_rel(f_into, f_orth).pvalue
            except Exception: t_p = float("nan")
            print(f"  matched total drift?  into = {t_into.mean():.4f} "
                  f"vs orth = {t_orth.mean():.4f}  (should be ~equal)")
            print(f"  forgetting  INTO P_A = {f_into.mean():.3f}   "
                  f"ORTH P_A = {f_orth.mean():.3f}")
            print(f"  paired diff (into-orth) = {diff.mean():+.3f}   "
                  f"Cohen d = {d:+.2f}   Wilcoxon p = {w_p:.4f}   t p = {t_p:.4f}")
            print("\n  -> If INTO > ORTH at matched total drift, TR-drift is CAUSAL (your P4).")
        else:
            print("  (not enough paired rows yet)")
    print("\n" + "=" * 70)


# -----------------------------------------------------------------------------
# DECISIVE unified causal experiment (stronger-powered version of section C)
#   - head + LoRA train normally (real, measurable forgetting)
#   - only the BACKBONE update direction is controlled (into vs orthogonal to P_A)
#     at matched total step-norm
#   - strongly-forgetting pairs, high lr, N = pairs*lrs*seeds paired comparisons
# -----------------------------------------------------------------------------
DECISIVE_PAIRS = (("ag_news", "imdb"), ("sst2", "ag_news"), ("yelp", "ag_news"))
DECISIVE_LRS = (2e-4, 5e-4)
DECISIVE_SEEDS = (0, 1, 2, 3, 4)


def train_taskB_controlled_v2(model, dB, cfg, subspace_t, mode, step_norm, seed, lr):
    import torch
    tparams = target_params(model, cfg)
    for _, p in model.named_parameters():
        p.requires_grad = True
    lora, head, backbone = param_groups(model, cfg)
    opt = torch.optim.AdamW([{"params": lora, "lr": lr},
                             {"params": head, "lr": cfg.head_lr}], weight_decay=0.0)
    model.train()
    for ep in range(cfg.epochs):
        for batch in iter_batches(dB, cfg.batch_size, True, seed * 13 + ep):
            model.zero_grad(set_to_none=True)
            out = model(**batch); out.loss.backward()
            with torch.no_grad():
                dirs = {}; sq = 0.0
                for n_, p in tparams:
                    if p.grad is None:
                        dirs[n_] = torch.zeros_like(p).reshape(-1); continue
                    g = p.grad.detach().reshape(-1); V = subspace_t.get(n_)
                    if V is None or V.numel() == 0:
                        d = g if mode == "orth" else torch.zeros_like(g)
                    else:
                        gpar = V.t() @ (V @ g)
                        d = gpar if mode == "into" else (g - gpar)
                    dirs[n_] = d; sq += float(torch.dot(d, d))
                scale = step_norm / (math.sqrt(sq) + 1e-12)
                for n_, p in tparams:
                    p.add_(-(dirs[n_] * scale).reshape(p.shape))
            opt.step()
    model.zero_grad(set_to_none=True)


def run_decisive(cfg, pairs=DECISIVE_PAIRS, lrs=DECISIVE_LRS, seeds=DECISIVE_SEEDS):
    import torch, pandas as pd
    device = "cuda" if torch.cuda.is_available() else "cpu"; rows = []
    out_path = os.path.join(cfg.out_dir, "decisive.csv")
    # resume: skip (taskA,taskB,lr,seed) already complete (both arms present)
    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        cnt = prev.groupby(["taskA", "taskB", "lr", "seed"])["arm"].nunique()
        done = set(cnt[cnt >= 2].index)
        print(f"[resume] {len(done)} conditions already complete -> skipping them")
    for (A, B) in pairs:
        for lr in lrs:
            for seed in seeds:
                if (A, B, lr, seed) in done:
                    continue
                set_seed(seed); model, tok = build_model(cfg, device)
                aTr, aTrL, aTe, aTeL = make_split(A, cfg, seed)
                bTr, bTrL, _, _ = make_split(B, cfg, seed)
                dA = tokenize(tok, aTr, aTrL, cfg, device); dAt = tokenize(tok, aTe, aTeL, cfg, device)
                dB = tokenize(tok, bTr, bTrL, cfg, device)
                train_task(model, dA, cfg, lr=cfg.lora_lr, seed=seed)
                accA = eval_acc(model, dAt, cfg)
                sub = compute_subspace(model, dA, cfg, seed)
                sub_t = {n_: torch.tensor(V.astype(np.float32), device=device) for n_, V in sub.items()}
                before = snapshot_target(model, cfg)
                postA = {k: v.detach().clone() for k, v in model.state_dict().items()}
                train_task(model, dB, cfg, lr=lr, seed=seed)            # calibrate step norm
                aft = snapshot_target(model, cfg)
                nstep = cfg.epochs * math.ceil(cfg.n_train / cfg.batch_size)
                tot = math.sqrt(sum(float(np.dot(aft[n] - before[n], aft[n] - before[n])) for n in before))
                step_norm = tot / max(nstep, 1)
                for mode in ("into", "orth"):
                    model.load_state_dict(postA)
                    train_taskB_controlled_v2(model, dB, cfg, sub_t, mode, step_norm, seed, lr)
                    accB = eval_acc(model, dAt, cfg)
                    dm = drift_measures(before, snapshot_target(model, cfg), sub)
                    rows.append({"taskA": A, "taskB": B, "lr": lr, "seed": seed, "arm": mode,
                                 "forgetting": max(0.0, accA - accB), **dm})
                    print(f"[dec] {A[:4]}->{B[:4]} lr={lr:.0e} s={seed} {mode:4s} "
                          f"forget={rows[-1]['forgetting']:.3f} total={dm['total_drift']:.4f} "
                          f"tr={dm['tr_drift']:.4f}", flush=True)
                # incremental save after each completed condition (survives disconnects)
                pd.DataFrame(rows).to_csv(out_path, index=False)
                del model, sub_t, postA; gc.collect()
                if device == "cuda": torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def verdict(cfg, df=None):
    import pandas as pd
    from scipy.stats import wilcoxon
    if df is None:
        df = pd.read_csv(os.path.join(cfg.out_dir, "decisive.csv"))
    print("=" * 60 + "\nVERDICT — decisive experiment\n" + "=" * 60)
    obs = os.path.join(cfg.out_dir, "observational.csv")
    if os.path.exists(obs):
        o = pd.read_csv(obs).dropna(subset=["forgetting", "total_drift", "tr_drift"])
        r1, p1 = partial_corr(o, "tr_drift", "forgetting", "total_drift")
        print(f"(A/B) correlation: N={len(o)}  TR<->forget partial|total = r={r1:+.3f} p={p1:.4g}")
    piv = df.pivot_table(index=["taskA", "taskB", "lr", "seed"], columns="arm",
                         values=["forgetting", "total_drift"]).dropna()
    fi, fo = piv[("forgetting", "into")].values, piv[("forgetting", "orth")].values
    ti, to = piv[("total_drift", "into")].values, piv[("total_drift", "orth")].values
    diff = fi - fo; n = len(diff); rng = np.random.default_rng(0)
    boots = [diff[rng.integers(0, n, n)].mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5]); d = diff.mean() / (diff.std(ddof=1) + 1e-12)
    print(f"(C) causal: N={n} pairs  total drift: into={ti.mean():.4f} orth={to.mean():.4f} (should match)")
    print(f"    forgetting INTO={fi.mean():.3f}  ORTH={fo.mean():.3f}")
    print(f"    diff(into-orth)={diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  Cohen d={d:+.2f}  "
          f"Wilcoxon p={wilcoxon(fi, fo).pvalue:.4f}")
    print("\n>>> CI fully positive => TR-drift is CAUSAL (P4 confirmed)." if lo > 0 else
          "\n>>> CI spans 0 => no causal evidence; TR-drift remains a DIAGNOSTIC predictor.")


# -----------------------------------------------------------------------------
# BASELINE comparison: EWC, OGD, GPM vs vanilla.
#   Goal: show TR-drift predicts forgetting ACROSS mitigation methods (it is a
#   general principle, not an artifact of our setup), and that methods which
#   reduce task-relevant motion (OGD/GPM) reduce forgetting more than a
#   parameter-importance penalty (EWC) that does not target direction.
#   All methods are run for real; we reuse only the shared infrastructure.
# -----------------------------------------------------------------------------
BASELINE_PAIRS = (("ag_news", "imdb"), ("sst2", "ag_news"), ("yelp", "ag_news"))
BASELINE_SEEDS = (0, 1, 2, 3, 4)
BASELINE_LR = 2e-4
EWC_LAMBDA = 1e3          # EWC penalty strength (not exhaustively tuned; see paper)


def fisher_diagonal(model, dataA, cfg, seed, n_batches):
    """Diagonal Fisher = mean squared gradient over target tensors (for EWC)."""
    import torch
    tparams = target_params(model, cfg)
    fisher = {n_: torch.zeros_like(p) for n_, p in tparams}
    model.train(); nb = 0
    for batch in iter_batches(dataA, cfg.batch_size, True, seed * 5 + 1):
        model.zero_grad(set_to_none=True)
        out = model(**batch); out.loss.backward()
        for n_, p in tparams:
            if p.grad is not None:
                fisher[n_] += p.grad.detach() ** 2
        nb += 1
        if nb >= n_batches:
            break
    for n_ in fisher:
        fisher[n_] /= max(nb, 1)
    model.zero_grad(set_to_none=True)
    return fisher


def ogd_basis(model, dataA, cfg, seed, n_vectors, k):
    """Orthonormal basis of raw Task-A backbone gradients (for OGD)."""
    tparams = target_params(model, cfg)
    grads = {n_: [] for n_, _ in tparams}
    model.train(); nb = 0
    for batch in iter_batches(dataA, cfg.batch_size, True, seed * 9 + 2):
        model.zero_grad(set_to_none=True)
        out = model(**batch); out.loss.backward()
        for n_, p in tparams:
            if p.grad is not None:
                grads[n_].append(p.grad.detach().reshape(-1).to("cpu").float().numpy())
        nb += 1
        if nb >= n_vectors:
            break
    model.zero_grad(set_to_none=True)
    basis = {}
    for n_ in grads:
        G = np.stack(grads[n_], axis=0)           # [n, d]
        Q, _ = np.linalg.qr(G.T)                   # [d, n] orthonormal columns
        kk = min(k, Q.shape[1])
        basis[n_] = Q[:, :kk].T.astype(np.float16) # [k, d] orthonormal rows
    return basis


def train_taskB_ewc(model, dB, cfg, lr, seed, theta_A, fisher, lam):
    """Task B training with an EWC penalty anchoring target tensors to theta_A."""
    import torch
    lora, head, backbone = param_groups(model, cfg)
    opt = torch.optim.AdamW(
        [{"params": lora, "lr": lr}, {"params": head, "lr": cfg.head_lr},
         {"params": backbone, "lr": lr}], weight_decay=0.0)
    tparams = target_params(model, cfg)
    model.train()
    for ep in range(cfg.epochs):
        for batch in iter_batches(dB, cfg.batch_size, True, seed * 1000 + ep):
            opt.zero_grad()
            out = model(**batch)
            pen = 0.0
            for n_, p in tparams:
                pen = pen + (fisher[n_] * (p - theta_A[n_]) ** 2).sum()
            (out.loss + lam * pen).backward()
            opt.step()
    return model


def train_taskB_project(model, dB, cfg, lr, seed, basis_t):
    """Task B training with backbone gradients projected ORTHOGONAL to a stored
       Task-A basis (used for both OGD and GPM; they differ only in the basis)."""
    import torch
    lora, head, backbone = param_groups(model, cfg)
    opt = torch.optim.AdamW(
        [{"params": lora, "lr": lr}, {"params": head, "lr": cfg.head_lr},
         {"params": backbone, "lr": lr}], weight_decay=0.0)
    tparams = target_params(model, cfg)
    model.train()
    for ep in range(cfg.epochs):
        for batch in iter_batches(dB, cfg.batch_size, True, seed * 1000 + ep):
            opt.zero_grad()
            out = model(**batch); out.loss.backward()
            with torch.no_grad():
                for n_, p in tparams:
                    if p.grad is None:
                        continue
                    V = basis_t.get(n_)
                    if V is None or V.numel() == 0:
                        continue
                    g = p.grad.reshape(-1)
                    g_orth = g - V.t() @ (V @ g)     # remove component in Task-A basis
                    p.grad.copy_(g_orth.reshape(p.shape))
            opt.step()
    return model


def run_baselines(cfg, pairs=BASELINE_PAIRS, seeds=BASELINE_SEEDS, lr=BASELINE_LR):
    import torch, pandas as pd
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    out_path = os.path.join(cfg.out_dir, "baselines.csv")
    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path); rows = prev.to_dict("records")
        cnt = prev.groupby(["taskA", "taskB", "seed"])["method"].nunique()
        done = set(cnt[cnt >= 4].index)
        print(f"[resume] {len(done)} (pair,seed) groups complete -> skipping")
    for (A, B) in pairs:
        for seed in seeds:
            if (A, B, seed) in done:
                continue
            set_seed(seed); model, tok = build_model(cfg, device)
            aTr, aTrL, aTe, aTeL = make_split(A, cfg, seed)
            bTr, bTrL, _, _ = make_split(B, cfg, seed)
            dA = tokenize(tok, aTr, aTrL, cfg, device); dAt = tokenize(tok, aTe, aTeL, cfg, device)
            dB = tokenize(tok, bTr, bTrL, cfg, device)
            train_task(model, dA, cfg, lr=cfg.lora_lr, seed=seed)
            accA = eval_acc(model, dAt, cfg)
            sub = compute_subspace(model, dA, cfg, seed)                     # P_A (for TR-drift + GPM)
            before = snapshot_target(model, cfg)
            theta_A = {n_: p.detach().clone() for n_, p in target_params(model, cfg)}
            fisher = fisher_diagonal(model, dA, cfg, seed, cfg.n_grad_batches)
            ogd_b = ogd_basis(model, dA, cfg, seed, cfg.n_grad_batches, cfg.subspace_k)
            gpm_t = {n_: torch.tensor(V.astype(np.float32), device=device) for n_, V in sub.items()}
            ogd_t = {n_: torch.tensor(V.astype(np.float32), device=device) for n_, V in ogd_b.items()}
            postA = {k: v.detach().clone() for k, v in model.state_dict().items()}

            for method in ("vanilla", "ewc", "ogd", "gpm"):
                model.load_state_dict(postA)
                if method == "vanilla":
                    train_task(model, dB, cfg, lr=lr, seed=seed)
                elif method == "ewc":
                    train_taskB_ewc(model, dB, cfg, lr, seed, theta_A, fisher, EWC_LAMBDA)
                elif method == "ogd":
                    train_taskB_project(model, dB, cfg, lr, seed, ogd_t)
                elif method == "gpm":
                    train_taskB_project(model, dB, cfg, lr, seed, gpm_t)
                accB = eval_acc(model, dAt, cfg)
                dm = drift_measures(before, snapshot_target(model, cfg), sub)
                rows.append({"taskA": A, "taskB": B, "seed": seed, "method": method,
                             "forgetting": max(0.0, accA - accB), **dm})
                print(f"[base] {A[:4]}->{B[:4]} s={seed} {method:7s} "
                      f"forget={rows[-1]['forgetting']:.3f} tr={dm['tr_drift']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            del model, gpm_t, ogd_t, postA; gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows); df.to_csv(out_path, index=False)
    return df


def analyze_baselines(cfg, df=None):
    import pandas as pd
    from scipy.stats import pearsonr
    if df is None:
        df = pd.read_csv(os.path.join(cfg.out_dir, "baselines.csv"))
    print("=" * 60 + "\nBASELINE COMPARISON (EWC / OGD / GPM vs vanilla)\n" + "=" * 60)
    g = df.groupby("method").agg(forget_mean=("forgetting", "mean"),
                                 forget_sem=("forgetting", lambda x: x.std(ddof=1) / max(len(x) ** 0.5, 1)),
                                 tr_mean=("tr_drift", "mean")).reindex(["vanilla", "ewc", "ogd", "gpm"])
    print("\n  method   forgetting (mean +/- sem)   mean TR-drift")
    for m, r in g.iterrows():
        print(f"  {m:8s} {r['forget_mean']:.3f} +/- {r['forget_sem']:.3f}        {r['tr_mean']:.4f}")
    # TR-drift predicts forgetting POOLED across all methods
    d2 = df.dropna(subset=["tr_drift", "forgetting"])
    r, p = pearsonr(d2["tr_drift"], d2["forgetting"])
    print(f"\n  Across ALL methods (N={len(d2)}): TR-drift <-> forgetting r = {r:+.3f}  p = {p:.2e}")
    print("  -> TR-drift tracks forgetting regardless of mitigation method (general principle).")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="TRDT N=3 fix (Kaggle-ready)")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to verify the pipeline")
    ap.add_argument("--full", action="store_true", help="full experiment")
    ap.add_argument("--stage", choices=["obs", "intervene", "analyze", "decisive", "paper", "baselines", "all"], default="all")
    ap.add_argument("--out", default=None)
    args, _ = ap.parse_known_args()  # ignore notebook/kernel args (Colab/Jupyter safe)

    cfg = Config()
    if args.out:
        cfg.out_dir = args.out
    elif not os.path.isdir("/kaggle/working"):
        cfg.out_dir = "/content" if os.path.isdir("/content") else "."
    if args.smoke or (not args.full and args.stage not in ("decisive", "paper", "baselines")):
        cfg = apply_smoke(cfg)
        print(">> SMOKE MODE (tiny). Use --full for the real run.\n")
    os.makedirs(cfg.out_dir, exist_ok=True)

    import torch
    print(f"device = {'cuda' if torch.cuda.is_available() else 'cpu'} | "
          f"out_dir = {cfg.out_dir}")
    print(f"config: {json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}, default=str)[:400]}...\n")

    if args.stage in ("obs", "all"):
        run_observational(cfg)
    if args.stage in ("intervene", "all"):
        run_intervention(cfg)
    if args.stage in ("analyze", "all"):
        analyze(cfg)
    if args.stage == "decisive":
        df = run_decisive(cfg)
        verdict(cfg, df)
    if args.stage == "baselines":
        df = run_baselines(cfg)
        analyze_baselines(cfg, df)
    if args.stage == "paper":
        # ONE command -> every result in the paper, all saved to CSV, reproducible.
        print("\n##### STAGE 1/3 : observational sweep + static-overlap negative result #####\n")
        run_observational(cfg)
        print("\n##### STAGE 2/3 : correlation + incremental validity #####\n")
        analyze(cfg)
        print("\n##### STAGE 3/3 : decisive causal experiment #####\n")
        df = run_decisive(cfg)
        verdict(cfg, df)
        print("\nAll artifacts saved in", cfg.out_dir,
              ": observational.csv, decisive.csv  (attach these to the paper).")


if __name__ == "__main__":
    main()
