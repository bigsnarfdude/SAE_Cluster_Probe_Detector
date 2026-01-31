#!/usr/bin/env python3
"""
Exp 11d: Fine-grained L1 sweep around best C=0.05, and explore:
1. Fine L1 sweep on L40: C in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
2. L1 on L35_40 concat with moderate max_iter
3. Cross-validation on training set to check stability
4. Export the best probe's weight vector for neuron-level interpretation
"""

import json, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

OUTPUT_DIR = Path("/home/ubuntu/results_11c")  # reuse cached states
RESULTS_DIR = Path("/home/ubuntu/results_11d")
RESULTS_DIR.mkdir(exist_ok=True)

def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"roc_auc": float(roc_auc_score(y_true, y_prob)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "tpr": float(tp/(tp+fn)) if (tp+fn)>0 else 0,
            "fpr": float(fp/(fp+tn)) if (fp+tn)>0 else 0,
            "tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn)}

# Load cached states
s = np.load(OUTPUT_DIR / "full_states_11c.npz", allow_pickle=True)
af_s = s["af_states"].item()
nonaf_s = s["nonaf_states"].item()
gaf_s = s["gold_af_states"].item()
gal_s = s["gold_aligned_states"].item()

na = len(af_s[40])
nn = len(nonaf_s[40])
nga = len(gaf_s[40])
ngl = len(gal_s[40])
yg = np.array([1]*nga + [0]*ngl)
print(f"Train: {na} AF + {nn} non-AF, Gold: {nga} AF + {ngl} aligned")

results = {}

# ============================================================
# TEST 1: Fine L1 sweep on L40
# ============================================================
print("\nTEST 1: Fine L1 sweep on L40")
Xtr = np.vstack([af_s[40], nonaf_s[40]])
ytr = np.array([1]*na + [0]*nn)
Xte = np.vstack([gaf_s[40], gal_s[40]])

sc = StandardScaler()
Xtr_s = sc.fit_transform(Xtr)
Xte_s = sc.transform(Xte)

fine_l1 = {}
for C in [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.08, 0.09]:
    lr = LogisticRegression(C=C, penalty="l1", solver="saga",
                            class_weight="balanced", max_iter=5000, random_state=42)
    lr.fit(Xtr_s, ytr)
    yp = lr.predict(Xte_s)
    yb = lr.predict_proba(Xte_s)[:, 1]
    m = compute_metrics(yg, yp, yb)
    nz = int(np.sum(np.abs(lr.coef_[0]) > 1e-10))
    m["n_nonzero_weights"] = nz
    key = "l1_C" + str(C)
    fine_l1[key] = m
    print(f"  {key}: AUC={m['roc_auc']:.3f} FPR={m['fpr']:.3f} nz={nz}")

    # Save sparse model for the best
    if nz > 0 and nz < 300:
        nz_idx = np.where(np.abs(lr.coef_[0]) > 1e-10)[0]
        nz_weights = lr.coef_[0][nz_idx]
        np.savez(RESULTS_DIR / f"sparse_l1_C{C}.npz",
                 indices=nz_idx, weights=nz_weights,
                 intercept=np.array([lr.intercept_[0]]),
                 scaler_mean=sc.mean_, scaler_scale=sc.scale_)

results["fine_l1_L40"] = fine_l1

# ============================================================
# TEST 2: Cross-validation stability
# ============================================================
print("\nTEST 2: 5-fold CV on training set (C=0.05)")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = []
cv_nz = []
for fold, (train_idx, val_idx) in enumerate(skf.split(Xtr_s, ytr)):
    lr = LogisticRegression(C=0.05, penalty="l1", solver="saga",
                            class_weight="balanced", max_iter=5000, random_state=42)
    lr.fit(Xtr_s[train_idx], ytr[train_idx])
    yb = lr.predict_proba(Xtr_s[val_idx])[:, 1]
    auc = roc_auc_score(ytr[val_idx], yb)
    nz = int(np.sum(np.abs(lr.coef_[0]) > 1e-10))
    cv_aucs.append(auc)
    cv_nz.append(nz)
    print(f"  Fold {fold+1}: AUC={auc:.3f} nz={nz}")

results["cv_stability"] = {
    "cv_aucs": cv_aucs,
    "cv_mean_auc": float(np.mean(cv_aucs)),
    "cv_std_auc": float(np.std(cv_aucs)),
    "cv_nz": cv_nz,
    "cv_mean_nz": float(np.mean(cv_nz)),
}
print(f"  Mean AUC: {np.mean(cv_aucs):.3f} +/- {np.std(cv_aucs):.3f}")

# ============================================================
# TEST 3: Best model weight analysis
# ============================================================
print("\nTEST 3: Best model weight analysis")
# Re-fit the best model
best_C = 0.05
lr_best = LogisticRegression(C=best_C, penalty="l1", solver="saga",
                              class_weight="balanced", max_iter=5000, random_state=42)
lr_best.fit(Xtr_s, ytr)
coef = lr_best.coef_[0]
nz_mask = np.abs(coef) > 1e-10
nz_idx = np.where(nz_mask)[0]
nz_w = coef[nz_idx]

# Save full weight vector (useful for dim-level analysis)
np.savez(RESULTS_DIR / "best_probe_weights.npz",
         full_coef=coef, intercept=np.array([lr_best.intercept_[0]]),
         nonzero_indices=nz_idx, nonzero_weights=nz_w,
         scaler_mean=sc.mean_, scaler_scale=sc.scale_)

# Top positive (AF-indicating) and negative (aligned-indicating) dimensions
sorted_pos = np.argsort(-nz_w)  # most positive first
sorted_neg = np.argsort(nz_w)   # most negative first

top_af_dims = []
for i in sorted_pos[:20]:
    if nz_w[i] > 0:
        top_af_dims.append({"dim": int(nz_idx[i]), "weight": float(nz_w[i])})

top_aligned_dims = []
for i in sorted_neg[:20]:
    if nz_w[i] < 0:
        top_aligned_dims.append({"dim": int(nz_idx[i]), "weight": float(nz_w[i])})

results["best_model"] = {
    "C": best_C,
    "n_nonzero": int(len(nz_idx)),
    "intercept": float(lr_best.intercept_[0]),
    "top_af_dimensions": top_af_dims,
    "top_aligned_dimensions": top_aligned_dims,
    "roc_auc_gold106": float(roc_auc_score(yg, lr_best.predict_proba(Xte_s)[:, 1])),
}
print(f"  Best model: {len(nz_idx)} nonzero dims, AUC={results['best_model']['roc_auc_gold106']:.3f}")
print(f"  Top AF dims: {[d['dim'] for d in top_af_dims[:5]]}")
print(f"  Top aligned dims: {[d['dim'] for d in top_aligned_dims[:5]]}")

# ============================================================
# TEST 4: Ablation - how many top dims do you need?
# ============================================================
print("\nTEST 4: Top-k dimension ablation")
ablation = {}
sorted_by_abs = np.argsort(-np.abs(nz_w))
for k in [5, 10, 20, 30, 50, 80, 100, 150, 174]:
    if k > len(nz_idx):
        break
    top_k_idx = nz_idx[sorted_by_abs[:k]]
    # Use only these dimensions
    Xtr_k = Xtr_s[:, top_k_idx]
    Xte_k = Xte_s[:, top_k_idx]
    lr_k = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
    lr_k.fit(Xtr_k, ytr)
    yb_k = lr_k.predict_proba(Xte_k)[:, 1]
    yp_k = lr_k.predict(Xte_k)
    m_k = compute_metrics(yg, yp_k, yb_k)
    ablation["top_" + str(k)] = m_k
    print(f"  top-{k}: AUC={m_k['roc_auc']:.3f} FPR={m_k['fpr']:.3f}")

results["ablation_topk"] = ablation

# ============================================================
# TEST 5: L1 on L35_40 concatenated (with capped iterations)
# ============================================================
print("\nTEST 5: L1 on L35_40 concatenated")
Xtr_m = np.vstack([np.hstack([af_s[35], af_s[40]]), np.hstack([nonaf_s[35], nonaf_s[40]])])
Xte_m = np.vstack([np.hstack([gaf_s[35], gaf_s[40]]), np.hstack([gal_s[35], gal_s[40]])])
sc_m = StandardScaler()
Xtr_ms = sc_m.fit_transform(Xtr_m)
Xte_ms = sc_m.transform(Xte_m)

multi_l1 = {}
for C in [0.01, 0.03, 0.05, 0.07, 0.1]:
    lr = LogisticRegression(C=C, penalty="l1", solver="saga",
                            class_weight="balanced", max_iter=2000, random_state=42)
    lr.fit(Xtr_ms, ytr)
    yp = lr.predict(Xte_ms)
    yb = lr.predict_proba(Xte_ms)[:, 1]
    m = compute_metrics(yg, yp, yb)
    nz = int(np.sum(np.abs(lr.coef_[0]) > 1e-10))
    m["n_nonzero_weights"] = nz
    key = "l1_C" + str(C)
    multi_l1[key] = m
    print(f"  L35_40 {key}: AUC={m['roc_auc']:.3f} FPR={m['fpr']:.3f} nz={nz}")

results["l1_L35_40"] = multi_l1

# Save
with open(RESULTS_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results to", RESULTS_DIR / "results.json")
