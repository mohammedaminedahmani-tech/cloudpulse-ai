"""Pôle 2 - Fine-tuning efficace (T2) : full FT, LoRA, QLoRA sur une tâche.

Modèle : Qwen2.5-0.5B-Instruct + tête de classification (2 classes).
Mesure : F1 macro, accuracy, rappel anomalie, paramètres entraînables,
         pic mémoire GPU, temps d'entraînement, taille sur disque.
Résultats ajoutés dans docs/pole2/resultats_finetuning.csv

Usage (GPU requis) :
    python -m src.logs.finetune --mode lora --task A --seeds 0 1 2
    modes : full | lora | qlora
"""
import argparse
import gc
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig, get_linear_schedule_with_warmup)

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = Path("data/logs")
OUT = Path("docs/pole2")
MODELS = Path("models/pole2")
LR = {"full": 2e-5, "lora": 2e-4, "qlora": 2e-4}


def load_split(task: str, part: str):
    df = pd.read_csv(DATA / f"task_{task}_{part}.csv")
    return df["text"].fillna("").astype(str).tolist(), df["label"].to_numpy()


def build_model(mode: str, tok):
    if mode == "qlora":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE, num_labels=2, quantization_config=bnb, device_map={"": 0})
        model.config.pad_token_id = tok.pad_token_id
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE, num_labels=2, torch_dtype=torch.float32).to("cuda")
        model.config.pad_token_id = tok.pad_token_id
    if mode in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05,
                         target_modules=["q_proj", "v_proj"], task_type="SEQ_CLS")
        model = get_peft_model(model, cfg)
    else:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


@torch.no_grad()
def predict(model, tok, texts, max_len):
    model.eval()
    out = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(**enc).logits.float().softmax(-1).cpu())
    return torch.cat(out).numpy()


def metrics(y, probs):
    pred = probs.argmax(1)
    return {"f1_macro": f1_score(y, pred, average="macro"),
            "accuracy": accuracy_score(y, pred),
            "rappel_anomalie": recall_score(y, pred, pos_label=1)}


def run(mode, task, seed, epochs, bs, max_len):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token

    Xtr, ytr = load_split(task, "train")
    Xva, yva = load_split(task, "val")
    Xte, yte = load_split(task, "test")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(mode, tok)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[{mode} s{seed}] paramètres entraînables : {n_train:,} / {n_total:,} "
          f"({100 * n_train / n_total:.2f} %)")

    opt = torch.optim.AdamW(params, lr=LR[mode], weight_decay=0.01)
    steps = epochs * math.ceil(len(Xtr) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(Xtr))
        total_loss = 0.0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([Xtr[j] for j in idx], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to("cuda")
            y = torch.tensor(ytr[idx], device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**enc).logits
            loss = F.cross_entropy(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            total_loss += loss.item() * len(idx)
        print(f"   époque {ep + 1}/{epochs} : perte {total_loss / len(Xtr):.4f}", flush=True)
    train_time = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    res_val = metrics(yva, predict(model, tok, Xva, max_len))
    res_test = metrics(yte, predict(model, tok, Xte, max_len))

    if mode == "full":
        size_mb = n_total * 4 / 1e6          # estimation : poids fp32 (non sauvegardé)
    else:
        path = MODELS / f"{mode}_{task}_s{seed}"
        model.save_pretrained(path)
        size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6

    row = {"mode": mode, "tache": task, "graine": seed, "epoques": epochs,
           "params_entrainables": n_train, "pct_entrainable": round(100 * n_train / n_total, 3),
           "pic_memoire_gpu_Go": round(peak_gb, 2), "temps_s": round(train_time, 1),
           "taille_Mo": round(size_mb, 1),
           "f1_val": round(res_val["f1_macro"], 4),
           "f1_test": round(res_test["f1_macro"], 4),
           "accuracy_test": round(res_test["accuracy"], 4),
           "rappel_test": round(res_test["rappel_anomalie"], 4)}
    print(row)
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "lora", "qlora"], required=True)
    ap.add_argument("--task", default="A")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("GPU requis : Colab > Exécution > Modifier le type d'exécution > T4.")
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([run(a.mode, a.task, s, a.epochs, a.bs, a.max_len) for s in a.seeds])
    f = OUT / "resultats_finetuning.csv"
    if f.exists():
        new = pd.concat([pd.read_csv(f), new], ignore_index=True)
    new.to_csv(f, index=False)


if __name__ == "__main__":
    main()