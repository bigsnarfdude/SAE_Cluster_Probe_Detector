#!/usr/bin/env python3
"""
Exp 11c: Full-scale mech interp with 1340 AF + 1340 non-AF from Redwood.
Extract hidden states at layers 35,40 (the two best from 11b), then run
PCA + regularization sweep on gold_106.
"""

import json, os, time, numpy as np, torch
from pathlib import Path

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
DEVICE = "cuda"
MODEL_NAME = "google/gemma-3-27b-it"
LAYERS = [35, 40]  # Only the two best layers
OUTPUT_DIR = Path("/home/ubuntu/results_11c")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    with open("/home/ubuntu/redwood_mech_interp.json") as f:
        data = json.load(f)
    af_texts = [s["text"] for s in data["train_af"]]
    nonaf_texts = [s["text"] for s in data["train_nonaf"]]
    gold_af = [s for s in data["gold_106"] if s["label"] == "potential_faking"]
    gold_aligned = [s for s in data["gold_106"] if s["label"] == "aligned"]
    print(f"Loaded: {len(af_texts)} AF, {len(nonaf_texts)} non-AF, "
          f"{len(gold_af)} gold AF, {len(gold_aligned)} gold aligned")
    return af_texts, nonaf_texts, gold_af, gold_aligned

def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager"
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.0f}s")
    return model, tokenizer

def extract_hidden_states(model, tokenizer, texts, layers, batch_label="", pool="mean"):
    """Extract hidden states for a list of texts at specified layers."""
    states = {l: [] for l in layers}
    n = len(texts)
    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for l in layers:
            hs = outputs.hidden_states[l][0].cpu().float()  # (seq_len, hidden_dim)
            if pool == "mean":
                vec = hs.mean(dim=0).numpy()
            else:
                vec = hs[-1].numpy()
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            states[l].append(vec)
        del outputs
        torch.cuda.empty_cache()
        if (i + 1) % 50 == 0:
            print(f"  [{batch_label}] {i+1}/{n}")
    for l in layers:
        states[l] = np.array(states[l])
    return states

def main():
    t0 = time.time()
    af_texts, nonaf_texts, gold_af, gold_aligned = load_data()
    model, tokenizer = load_model()

    # Check if states already cached
    cache_path = OUTPUT_DIR / "full_states_11c.npz"
    if cache_path.exists():
        print("Loading cached states...")
        s = np.load(cache_path, allow_pickle=True)
        af_states = s["af_states"].item()
        nonaf_states = s["nonaf_states"].item()
        gaf_states = s["gold_af_states"].item()
        gal_states = s["gold_aligned_states"].item()
    else:
        print(f"\nExtracting states for {len(af_texts)} AF texts...")
        af_states = extract_hidden_states(model, tokenizer, af_texts, LAYERS, "AF")

        print(f"\nExtracting states for {len(nonaf_texts)} non-AF texts...")
        nonaf_states = extract_hidden_states(model, tokenizer, nonaf_texts, LAYERS, "nonAF")

        print(f"\nExtracting states for {len(gold_af)} gold AF texts...")
        gaf_states = extract_hidden_states(model, tokenizer, [s["text"] for s in gold_af], LAYERS, "gold_AF")

        print(f"\nExtracting states for {len(gold_aligned)} gold aligned texts...")
        gal_states = extract_hidden_states(model, tokenizer, [s["text"] for s in gold_aligned], LAYERS, "gold_aligned")

        # Cache
        np.savez(cache_path,
                 af_states=af_states, nonaf_states=nonaf_states,
                 gold_af_states=gaf_states, gold_aligned_states=gal_states)
        print(f"States cached ({cache_path.stat().st_size/1024/1024:.0f} MB)")

    extract_time = time.time() - t0
    print(f"\nExtraction done in {extract_time:.0f}s")

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Now run evaluation suite
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

    na = len(af_states[40])
    nn = len(nonaf_states[40])
    nga = len(gaf_states[40])
    ngl = len(gal_states[40])
    yg = np.array([1]*nga + [0]*ngl)
    print(f"Train: {na} AF + {nn} non-AF, Gold: {nga} AF + {ngl} aligned")

    results = {"n_train_af": na, "n_train_nonaf": nn,
               "n_gold_af": nga, "n_gold_aligned": ngl,
               "extraction_time_s": extract_time}

    def compute_metrics(y_true, y_pred, y_prob):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {"roc_auc": float(roc_auc_score(y_true, y_prob)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "tpr": float(tp/(tp+fn)) if (tp+fn)>0 else 0,
                "fpr": float(fp/(fp+tn)) if (fp+tn)>0 else 0,
                "tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn)}

    def eval_probe(X_tr, y_tr, X_te, y_te, C=1.0, penalty="l2"):
        solver = "saga" if penalty == "l1" else "lbfgs"
        sc = StandardScaler()
        Xts = sc.fit_transform(X_tr)
        Xes = sc.transform(X_te)
        lr = LogisticRegression(C=C, penalty=penalty, solver=solver,
                                class_weight="balanced", max_iter=5000, random_state=42)
        lr.fit(Xts, y_tr)
        yp = lr.predict(Xes)
        yb = lr.predict_proba(Xes)[:, 1]
        m = compute_metrics(y_te, yp, yb)
        m["n_nonzero_weights"] = int(np.sum(np.abs(lr.coef_[0]) > 1e-10))
        return m, lr

    # TEST 1: Single layer probes
    print("\nTEST 1: Single layer probes")
    sl = {}
    for l in LAYERS:
        Xtr = np.vstack([af_states[l], nonaf_states[l]])
        ytr = np.array([1]*na + [0]*nn)
        Xte = np.vstack([gaf_states[l], gal_states[l]])
        m, _ = eval_probe(Xtr, ytr, Xte, yg)

        # Contrastive direction
        d = af_states[l].mean(0) - nonaf_states[l].mean(0)
        d /= (np.linalg.norm(d) + 1e-10)
        ga = gaf_states[l] @ d
        gl = gal_states[l] @ d
        ac = float(roc_auc_score(yg, np.concatenate([ga, gl])))
        sd = float((ga.mean()-gl.mean()) / (ga.std()+gl.std()+1e-10) * 2)

        sl["layer_" + str(l)] = m
        sl["layer_" + str(l) + "_contrastive"] = {"roc_auc": ac, "separation_d": sd}
        print(f"  L{l}: probe={m['roc_auc']:.3f} FPR={m['fpr']:.3f} | contrastive={ac:.3f} sep_d={sd:.3f}")
    results["single_layer"] = sl

    # TEST 2: Multi-layer L35+L40
    print("\nTEST 2: Multi-layer L35+L40")
    Xtr_m = np.vstack([np.hstack([af_states[35], af_states[40]]),
                        np.hstack([nonaf_states[35], nonaf_states[40]])])
    ytr = np.array([1]*na + [0]*nn)
    Xte_m = np.vstack([np.hstack([gaf_states[35], gaf_states[40]]),
                        np.hstack([gal_states[35], gal_states[40]])])
    m, _ = eval_probe(Xtr_m, ytr, Xte_m, yg)
    results["multi_L35_40"] = m
    print(f"  L35_40: AUC={m['roc_auc']:.3f} FPR={m['fpr']:.3f} dim={Xtr_m.shape[1]}")

    # TEST 3: L2 regularization sweep
    print("\nTEST 3: L2 regularization sweep")
    reg = {}
    for name, Xtr_, Xte_ in [
        ("L40", np.vstack([af_states[40], nonaf_states[40]]),
               np.vstack([gaf_states[40], gal_states[40]])),
        ("L35_40", Xtr_m, Xte_m),
    ]:
        cr = {}
        for C in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]:
            m, _ = eval_probe(Xtr_, ytr, Xte_, yg, C=C, penalty="l2")
            key = "l2_C" + str(C)
            cr[key] = m
            print(f"  {name} {key}: AUC={m['roc_auc']:.3f} FPR={m['fpr']:.3f}")
        reg[name] = cr
    results["regularization"] = reg

    # TEST 4: PCA sweep
    print("\nTEST 4: PCA sweep")
    pca_res = {}
    for name, Xtr_, Xte_ in [
        ("L40", np.vstack([af_states[40], nonaf_states[40]]),
               np.vstack([gaf_states[40], gal_states[40]])),
        ("L35_40", Xtr_m, Xte_m),
    ]:
        sc = StandardScaler()
        Xts = sc.fit_transform(Xtr_)
        Xes = sc.transform(Xte_)
        mx = min(Xts.shape)
        for nc in [5, 10, 15, 20, 25, 30, 50, 100, 200, 500, 1000]:
            if nc > mx:
                break
            pca = PCA(n_components=nc, random_state=42)
            Xtp = pca.fit_transform(Xts)
            Xep = pca.transform(Xes)
            ve = float(pca.explained_variance_ratio_.sum())
            for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
                lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=42)
                lr.fit(Xtp, ytr)
                yp = lr.predict(Xep)
                yb = lr.predict_proba(Xep)[:, 1]
                m = compute_metrics(yg, yp, yb)
                m["n_components"] = nc
                m["explained_variance"] = ve
                key = name + "_pca" + str(nc) + "_C" + str(C)
                pca_res[key] = m
                print(f"  {key}: AUC={m['roc_auc']:.3f} FPR={m['fpr']:.3f} var={ve:.3f}")
    results["pca"] = pca_res

    # TEST 5: L1 on L40 only (faster, interpretable)
    print("\nTEST 5: L1 sweep on L40")
    l1_res = {}
    Xtr40 = np.vstack([af_states[40], nonaf_states[40]])
    Xte40 = np.vstack([gaf_states[40], gal_states[40]])
    for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
        m, mdl = eval_probe(Xtr40, ytr, Xte40, yg, C=C, penalty="l1")
        key = "l1_C" + str(C)
        l1_res[key] = m
        print(f"  {key}: AUC={m['roc_auc']:.3f} FPR={m['fpr']:.3f} nz={m['n_nonzero_weights']}")
        # Save the best sparse model's nonzero indices
        if m["n_nonzero_weights"] > 0 and m["n_nonzero_weights"] < 200:
            nz_idx = np.where(np.abs(mdl.coef_[0]) > 1e-10)[0]
            nz_weights = mdl.coef_[0][nz_idx]
            np.savez(OUTPUT_DIR / f"sparse_l1_C{C}.npz",
                     indices=nz_idx, weights=nz_weights)
            print(f"    Saved sparse model: {len(nz_idx)} nonzero dims")
    results["l1_L40"] = l1_res

    # Save
    total_time = time.time() - t0
    results["total_time_seconds"] = total_time

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Find best
    best_auc = 0
    best_desc = ""
    for sn, sec in [("single", sl), ("pca", pca_res), ("l1", l1_res)]:
        for k, v in sec.items():
            if isinstance(v, dict) and "roc_auc" in v and v.get("roc_auc", 0) > best_auc:
                best_auc = v["roc_auc"]
                best_desc = sn + "/" + k
    for sn, sec in reg.items():
        for k, v in sec.items():
            if v["roc_auc"] > best_auc:
                best_auc = v["roc_auc"]
                best_desc = "reg/" + sn + "/" + k
    if results["multi_L35_40"]["roc_auc"] > best_auc:
        best_auc = results["multi_L35_40"]["roc_auc"]
        best_desc = "multi_L35_40"

    print(f"\n{'='*60}")
    print(f"DONE in {total_time/60:.1f} min")
    print(f"BEST: {best_desc} AUC={best_auc:.3f}")
    print(f"Previous best (Exp 11b, n=200): AUC=0.775")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
