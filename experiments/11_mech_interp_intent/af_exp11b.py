#!/usr/bin/env python3
"""Fast: reload states, run key tests, save results."""
import json, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

OUTPUT_DIR = Path("/home/ubuntu/results_11b")
LAYERS = [15, 20, 25, 30, 35, 40]

def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"roc_auc": float(roc_auc_score(y_true, y_prob)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "tpr": float(tp/(tp+fn)) if (tp+fn)>0 else 0,
            "fpr": float(fp/(fp+tn)) if (fp+tn)>0 else 0,
            "tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn)}

def eval_probe(X_tr, y_tr, X_te, y_te, C=1.0, penalty="l2"):
    solver = "saga" if penalty=="l1" else "lbfgs"
    sc = StandardScaler()
    Xts = sc.fit_transform(X_tr)
    Xes = sc.transform(X_te)
    lr = LogisticRegression(C=C, penalty=penalty, solver=solver,
                            class_weight="balanced", max_iter=5000, random_state=42)
    lr.fit(Xts, y_tr)
    yp = lr.predict(Xes)
    yb = lr.predict_proba(Xes)[:,1]
    m = compute_metrics(y_te, yp, yb)
    m["n_nonzero_weights"] = int(np.sum(np.abs(lr.coef_[0])>1e-10))
    return m, lr

s = np.load(OUTPUT_DIR/"full_states.npz", allow_pickle=True)
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
results = {"n_train_af":na, "n_train_nonaf":nn, "n_gold_af":nga, "n_gold_aligned":ngl}

# TEST 1: Single layer
print("TEST 1: Single layer probes")
sl = {}
for l in LAYERS:
    Xtr = np.vstack([af_s[l], nonaf_s[l]])
    ytr = np.array([1]*na+[0]*nn)
    Xte = np.vstack([gaf_s[l], gal_s[l]])
    m, _ = eval_probe(Xtr, ytr, Xte, yg)
    d = af_s[l].mean(0)-nonaf_s[l].mean(0)
    d /= (np.linalg.norm(d)+1e-10)
    ga = gaf_s[l]@d
    gl = gal_s[l]@d
    ac = float(roc_auc_score(yg, np.concatenate([ga,gl])))
    sd = float((ga.mean()-gl.mean())/(ga.std()+gl.std()+1e-10)*2)
    sl["layer_" + str(l)] = m
    sl["layer_" + str(l) + "_contrastive"] = {"roc_auc":ac, "separation_d":sd}
    print(f"  L{l}: probe={m['roc_auc']:.3f} contrastive={ac:.3f} sep_d={sd:.3f}")
results["single_layer"] = sl

# TEST 2: Multi-layer
print("TEST 2: Multi-layer")
ml = {}
for name, combo in [("L35_40",[35,40]),("L25_30_35_40",[25,30,35,40]),
                     ("L20_25_30_35_40",[20,25,30,35,40]),("L15_20_25_30_35_40",LAYERS)]:
    Xtr = np.vstack([np.hstack([af_s[l] for l in combo]), np.hstack([nonaf_s[l] for l in combo])])
    ytr = np.array([1]*na+[0]*nn)
    Xte = np.vstack([np.hstack([gaf_s[l] for l in combo]), np.hstack([gal_s[l] for l in combo])])
    m, _ = eval_probe(Xtr, ytr, Xte, yg)
    ml[name] = m
    print(f"  {name}: AUC={m['roc_auc']:.3f} dim={Xtr.shape[1]}")
results["multi_layer"] = ml

# TEST 3: Reg sweep - single L40 (both L1/L2), multi L2 only
print("TEST 3: Regularization sweep")
reg = {}

# Single L40
Xtr = np.vstack([af_s[40], nonaf_s[40]])
ytr = np.array([1]*na+[0]*nn)
Xte = np.vstack([gaf_s[40], gal_s[40]])
cr = {}
for pen in ["l1","l2"]:
    for C in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        m, mdl = eval_probe(Xtr, ytr, Xte, yg, C=C, penalty=pen)
        key = pen + "_C" + str(C)
        cr[key] = m
        print(f"  single_L40 {key}: AUC={m['roc_auc']:.3f} FPR={m['fpr']:.3f} nz={m['n_nonzero_weights']}")
reg["single_L40"] = cr

# Multi L25_30_35_40: L2 only
layers4 = [25,30,35,40]
Xtr2 = np.vstack([np.hstack([af_s[l] for l in layers4]), np.hstack([nonaf_s[l] for l in layers4])])
Xte2 = np.vstack([np.hstack([gaf_s[l] for l in layers4]), np.hstack([gal_s[l] for l in layers4])])
cr2 = {}
for C in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
    m, _ = eval_probe(Xtr2, ytr, Xte2, yg, C=C, penalty="l2")
    key = "l2_C" + str(C)
    cr2[key] = m
    print(f"  multi_L25_30_35_40 {key}: AUC={m['roc_auc']:.3f} FPR={m['fpr']:.3f}")
reg["multi_L25_30_35_40"] = cr2
results["regularization"] = reg

# TEST 4: PCA on multi L25_30_35_40
print("TEST 4: PCA on multi-layer")
bm = [25,30,35,40]
Xaf = np.hstack([af_s[l] for l in bm])
Xnaf = np.hstack([nonaf_s[l] for l in bm])
Xtrm = np.vstack([Xaf, Xnaf])
ytr = np.array([1]*na+[0]*nn)
Xtem = np.vstack([np.hstack([gaf_s[l] for l in bm]), np.hstack([gal_s[l] for l in bm])])
sc = StandardScaler()
Xts = sc.fit_transform(Xtrm)
Xes = sc.transform(Xtem)
mx = min(Xtrm.shape)
pca_res = {}
for nc in [5, 10, 15, 20, 25, 30, 50, 100, 200, 300]:
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
        yb = lr.predict_proba(Xep)[:,1]
        m = compute_metrics(yg, yp, yb)
        m["n_components"] = nc
        m["explained_variance"] = ve
        key = "pca" + str(nc) + "_C" + str(C)
        pca_res[key] = m
        print(f"  {key}: AUC={m['roc_auc']:.3f} FPR={m['fpr']:.3f} var={ve:.3f}")
results["pca"] = pca_res

# TEST 4b: PCA on L40 only
print("TEST 4b: PCA on L40 only")
Xtrm40 = np.vstack([af_s[40], nonaf_s[40]])
Xtem40 = np.vstack([gaf_s[40], gal_s[40]])
sc40 = StandardScaler()
Xts40 = sc40.fit_transform(Xtrm40)
Xes40 = sc40.transform(Xtem40)
mx40 = min(Xtrm40.shape)
pca40_res = {}
for nc in [5, 10, 15, 20, 30, 50, 100, 200, 300]:
    if nc > mx40:
        break
    pca = PCA(n_components=nc, random_state=42)
    Xtp = pca.fit_transform(Xts40)
    Xep = pca.transform(Xes40)
    ve = float(pca.explained_variance_ratio_.sum())
    for C in [0.001, 0.01, 0.1, 1.0]:
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=42)
        lr.fit(Xtp, ytr)
        yp = lr.predict(Xep)
        yb = lr.predict_proba(Xep)[:,1]
        m = compute_metrics(yg, yp, yb)
        m["n_components"] = nc
        m["explained_variance"] = ve
        key = "L40_pca" + str(nc) + "_C" + str(C)
        pca40_res[key] = m
        print(f"  {key}: AUC={m['roc_auc']:.3f} FPR={m['fpr']:.3f} var={ve:.3f}")
results["pca_L40"] = pca40_res

with open(OUTPUT_DIR/"results.json","w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results.json")

# Best overall
best_auc = 0
best_desc = ""
all_sections = [("single",sl),("multi",ml),("pca_multi",pca_res),("pca_L40",pca40_res)]
for sn, sec in all_sections:
    for k, v in sec.items():
        if isinstance(v, dict) and "roc_auc" in v and v["roc_auc"] > best_auc:
            best_auc = v["roc_auc"]
            best_desc = sn + "/" + k
for sn, sec in reg.items():
    for k, v in sec.items():
        if v["roc_auc"] > best_auc:
            best_auc = v["roc_auc"]
            best_desc = "reg/" + sn + "/" + k
print(f"\nBEST: {best_desc} AUC={best_auc:.3f}")
print(f"Previous best (Exp 11): AUC=0.714")
