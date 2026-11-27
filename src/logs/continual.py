"""Pôle 2 - Apprentissage continu : tâche A (HDFS) puis tâche B (BGL), avec LoRA.

Méthodes : naif, ewc (balayage de lambda), replay, joint (borne supérieure).
Résultats : docs/pole2/resultats_continu.csv (une ligne par méthode/lambda/graine)

Usage (GPU requis) :
    python -m src.logs.continual --seeds 0
    python -m src.logs.continual --seeds 1 2
"""
import argparse
import gc
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.logs import metrics as cl
from src.logs.ewc import compute_fisher, ewc_penalty, restore, snapshot
from src.logs.finetune import BASE, LR, build_model, load_split
from src.logs.finetune import metrics as clf_metrics
from src.logs.finetune import predict

OUT = Path("docs/pole2")
TASKS = ["A", "B"]
NAN = float("nan")


def train(model, tok, X, y, epochs, bs, max_len, rng, ewc_state=None, lam=0.0, tag=""):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR["lora"], weight_decay=0.01)
    steps = epochs * math.ceil(len(X) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(X))
        ce_sum, pen_sum = 0.0, 0.0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([X[j] for j in idx], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to("cuda")
            yb = torch.tensor(y[idx], device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**enc).logits
            ce = F.cross_entropy(logits.float(), yb)
            loss = ce
            if ewc_state is not None and lam > 0:
                pen = ewc_penalty(model, ewc_state["fisher"], ewc_state["theta"], lam)
                loss = ce + pen
                pen_sum += pen.item() * len(idx)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            ce_sum += ce.item() * len(idx)
        print(f"   [{tag}] époque {ep + 1}/{epochs} : perte {ce_sum / len(X):.4f}"
              + (f" | pénalité EWC {pen_sum / len(X):.4f}" if pen_sum else ""), flush=True)


def evaluate(model, tok, data, max_len):
    out = {}
    for t in TASKS:
        for part in ("val", "test"):
            X, y = data[t][part]
            out[(t, part)] = clf_metrics(y, predict(model, tok, X, max_len))
    return out


def make_row(method, lam, seed, aA, aB):
    f1 = lambda d, t, p: d[(t, p)]["f1_macro"] if d else NAN
    R = [[f1(aA, "A", "test"), f1(aA, "B", "test")],
         [f1(aB, "A", "test"), f1(aB, "B", "test")]]
    has_A = aA is not None
    return {
        "methode": method, "lambda": lam, "graine": seed,
        "f1_A_apres_A": R[0][0], "f1_B_apres_A": R[0][1],
        "f1_A_apres_B": R[1][0], "f1_B_apres_B": R[1][1],
        "rappel_A_apres_B": aB[("A", "test")]["rappel_anomalie"],
        "oubli_A": cl.forgetting(R) if has_A else NAN,
        "bwt": cl.bwt(R) if has_A else NAN,
        "f1_moyen_final": cl.average_final(R),
        "f1_val_moyen_final": float(np.mean([aB[("A", "val")]["f1_macro"],
                                             aB[("B", "val")]["f1_macro"]])),
    }


def fresh_model(tok, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return build_model("lora", tok)


def run_seed(seed, args, data, tok):
    XA, yA = data["A"]["train"]
    XB, yB = data["B"]["train"]
    rows = []
    print(f"\n######## graine {seed} ########", flush=True)

    model = fresh_model(tok, seed)
    train(model, tok, XA, yA, args.epochs, args.bs, args.max_len,
          np.random.RandomState(seed), tag="A")
    aA = evaluate(model, tok, data, args.max_len)
    print(f"après A : F1 A={aA[('A', 'test')]['f1_macro']:.4f} | "
          f"F1 B={aA[('B', 'test')]['f1_macro']:.4f}")
    theta_A = snapshot(model)
    fisher = compute_fisher(model, tok, XA, n_samples=args.n_fisher,
                            max_len=args.max_len, seed=seed)
    allf = torch.cat([f.flatten() for f in fisher.values()])
    print(f"Fisher : moyenne {allf.mean():.3e} | max {allf.max():.3e} | "
          f"{allf.numel():,} paramètres")
    ewc_state = {"fisher": fisher, "theta": theta_A}

    def run_B(method, lam, X, y, state=None):
        restore(model, theta_A)
        train(model, tok, X, y, args.epochs, args.bs, args.max_len,
              np.random.RandomState(seed + 100), ewc_state=state, lam=lam,
              tag=f"{method} lam={lam:g}")
        aB = evaluate(model, tok, data, args.max_len)
        row = make_row(method, lam, seed, aA, aB)
        print(f"-> {method} lam={lam:g} : F1 A={row['f1_A_apres_B']:.4f} | "
              f"F1 B={row['f1_B_apres_B']:.4f} | oubli A={row['oubli_A']:.4f}", flush=True)
        rows.append(row)

    run_B("naif", 0.0, XB, yB)
    for lam in args.lambdas:
        run_B("ewc", float(lam), XB, yB, ewc_state)
    rng = np.random.RandomState(seed + 200)
    k = max(1, int(args.replay_frac * len(XA)))
    pick = rng.choice(len(XA), k, replace=False)
    run_B("replay", 0.0, XB + [XA[i] for i in pick],
          np.concatenate([yB, yA[pick]]))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if not args.skip_joint:
        model = fresh_model(tok, seed)
        train(model, tok, XA + XB, np.concatenate([yA, yB]), args.epochs, args.bs,
              args.max_len, np.random.RandomState(seed + 300), tag="joint")
        aJ = evaluate(model, tok, data, args.max_len)
        rows.append(make_row("joint", 0.0, seed, None, aJ))
        print(f"-> joint : F1 A={aJ[('A', 'test')]['f1_macro']:.4f} | "
              f"F1 B={aJ[('B', 'test')]['f1_macro']:.4f}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def save(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "resultats_continu.csv"
    new = pd.DataFrame(rows)
    if f.exists():
        new = pd.concat([pd.read_csv(f), new], ignore_index=True)
        new = new.drop_duplicates(subset=["methode", "lambda", "graine"], keep="last")
    new.to_csv(f, index=False)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[10, 100, 1e3, 1e4, 1e5])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--n_fisher", type=int, default=200)
    ap.add_argument("--replay_frac", type=float, default=0.1)
    ap.add_argument("--skip_joint", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("GPU requis : Colab > Exécution > Modifier le type d'exécution > T4.")

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token
    data = {t: {p: load_split(t, p) for p in ("train", "val", "test")} for t in TASKS}
    for seed in args.seeds:
        df = save(run_seed(seed, args, data, tok))   # sauvegarde après chaque graine

    cols = ["f1_A_apres_B", "f1_B_apres_B", "oubli_A", "f1_moyen_final", "f1_val_moyen_final"]
    print("\n=== Résumé (moyenne sur les graines) ===")
    print(df.groupby(["methode", "lambda"])[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()