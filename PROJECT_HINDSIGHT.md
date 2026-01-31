# PROJECT HINDSIGHT
### Alignment Faking Detection Research Retrospective
### Dec 29 2025 - Jan 30 2026 | BigsnarfDude

---

## AT A GLANCE

```
 32 days  |  7 repos  |  370+ commits  |  5 published models  |  2,330-sample benchmark
```

```
OUTCOME: Application rejected. Research valid. Central mystery cracked open.
```

---

## THE PLOT

```
  AUROC
  1.0 ─                                          ●●●●●●●● fine-tuned detector
      │                                        ●
  0.9 ─                                       ●
      │                                      ●
  0.8 ─                                    ○○○○ sparse probe (152 neurons)
      │
  0.7 ─
      │
  0.6 ─
      │
  0.5 ─ ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
      │ ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○
      │ ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬ SAE probe (flat at 0.47)
      └──┬───────────────────────────────┬──┬────── day
       Dec 29                          Jan 29  30
                                          ↑
                                      GROK POINT
                                      stop decomposing.
                                      listen to the geometry.
```

```
 ● fine-tuned detector    0.98 AUROC   opaque, 27B parameters, black box
 ○ sparse linear probe    0.795 AUROC  interpretable, 152 neurons, auditable
 ▬ SAE feature probe      0.47 AUROC   random — decomposition destroys the signal
```

---

## THE ARC (5 phases)

```
 PHASE 1           PHASE 2            PHASE 3              PHASE 4           PHASE 5
 DETECTION         INTERPRETABILITY   HONEST RECKONING     APPLICATION       BREAKTHROUGH
 Dec 29-Jan 3      Jan 3-Jan 11       Jan 11-Jan 22        Jan 12+           Jan 29-30
 ────────────      ────────────       ────────────         ────────────      ────────────
 "Can we catch     "Can we explain    "What actually       "Ship it,         "Where does
  it?"              why?"              works?"              warts & all"       it live?"

 Result:           Result:            Result:              Result:           Result:
 YES 0.98 AUROC    NO 0.47 AUROC      Fine-tuning works.   Rejected.         152 neurons.
                                      SAEs don't.                            0.795 AUROC.
```

---

## SCOREBOARD

### What Works (validated on held-out Gold-106)

```
 MODEL                        AUROC    STATUS
 ─────────────────────────     ─────    ──────
 Gemma 3 27B fine-tuned        0.980    Best detector built
 GPT-OSS-20B fine-tuned        0.964    Cross-architecture confirmation
 GPT-OSS-120B fine-tuned       0.956    Scale doesn't help much
 Claude Haiku 4.5 zero-shot   ~0.980    Already solved out of the box
 Anthropic GPT-4o baseline     0.920    The number to beat (from paper)
 Sparse 152-dim probe (L40)    0.795    Best INTERPRETABLE result
```

### What Fails

```
 METHOD                       AUROC    FAILURE MODE
 ─────────────────────────     ─────    ─────────────────────────────
 SAE feature sweep (Gemma)     0.470    Random. Captures vocab not intent.
 SAE feature sweep (Llama)     0.570    Near-random.
 SAE feature engineering       0.858    0% specificity on hard negatives.
 SAE steering (all methods)    <10%     Features correlational, not causal.
 DAG structural (Redwood)      0.586    AF and aligned are structurally identical.
 SAE gradient attribution      0.500    Random. Gradient signal doesn't transfer.
 DAG on model organisms        0.423    Worse than random. Distribution mismatch.
```

---

## TIMELINE: KEY MOMENTS

```
Dec 29 ─── mindreader repo created. First detector experiments begin.
  |
Dec 31 ─── GlassBox: layer scanning Gemma-27B. Layers 31, 40, 53 look promising.
  |
Jan 1  ─── Hard negative audit. Feature 136 ablation looks amazing (100% flip!)
  |        ...turns out to be a methodology bug. First lesson in self-deception.
  |
Jan 3  ─── THREE things happen:
  |        1. BoxGlass forked from GlassBox (git tracking starts)
  |        2. Lightbright SAE experiments begin
  |        3. Contrastive feature discovery: 60 features, 0.82 AUROC probe
  |        4. Causal experiments v1-v5 ALL FAIL (0-3% flip rate)
  |           Root cause: zero-shot classifier had 0-8% accuracy on aligned
  |
Jan 4  ─── Gemma-3-27B detector trained (LoRA). Published to HuggingFace.
  |        unsloth-mlx: MLX port so Llama detector runs on Mac.
  |
Jan 5  ─── DEFINITIVE CAUSAL TESTS (v6 reruns, proper methodology):
  |        Ablation: 5% flip. Patching: 3.3%. Steering: 0-7%.
  |        VERDICT: SAE features = correlational, NOT causal.
  |
Jan 6  ─── af-detection-benchmark v1.0.0 released on HuggingFace.
  |        1,560 samples, 5 classes. The "compliant_hard_negative" class
  |        will later prove to be the critical innovation.
  |
Jan 7  ─┬─ DATA LEAKAGE DISCOVERED.
  |      |  Original SAE result: 0.83 AUROC. Features were selected using
  |      |  test set performance. Honest result: 0.47 AUROC. Random.
  |      |  All documentation corrected. Negative result published.
  |      └─ This is the ethical inflection point of the project.
  |
Jan 11 ─── Benchmark quality review complete. 30 commits.
  |
Jan 12 ─── APPLICATION DEADLINE. Submitted to Anthropic Fellows.
  |        Hook: "I built a detector that works (0.96). Then I tried to
  |        understand why using SAE interpretability. That failed (0.47)."
  |
Jan 14 ─── mindreader README restructured. 87 commits total. Repo frozen.
  |
Jan 20 ─── Feature engineering attempt: commander experiment.
  |        0.858 AUROC on gold_106. Looks promising...
  |
Jan 21 ─── COMPLIANT HARD NEGATIVE TEST: 0% specificity.
  |        All 520 samples misclassified. 100% false positive rate.
  |        SAE features trigger on RLHF vocabulary regardless of intent.
  |        THE VOCABULARY CONFOUND IS FUNDAMENTAL.
  |
Jan 22 ─── Final Lightbright experiments. 122 commits total.
  |        Per-trace vs cross-trace theory documented.
  |        Ensemble probes, adversarial search: nothing saves SAEs.
  |
Jan 29 ─── Full backup created (237 MB). Project Hindsight begins.
  |        SAE_Cluster_Probe_Detector: systematic re-evaluation of all approaches.
  |        Exp 01-09: DAG structural features, contrastive steering, gradient
  |        attribution. Everything tested rigorously against gold_106.
  |        Every method fails to generalize.
  |
Jan 30 ─── PHASE 5: MECHANISTIC INTERPRETABILITY BREAKTHROUGH.
  |        Lambda H100 rented. Gemma-3-27B hidden states extracted for
  |        1,340 Redwood AF transcripts at layers 35-40.
  |
  |        Exp 11:  Linear probe on layer 40 → AUC=0.714 (first signal > 0.7)
  |        Exp 11b: PCA + regularization sweep → AUC=0.775
  |        Exp 11c: Scaled to 1,340 training samples, L1 → AUC=0.791
  |        Exp 11d: Fine sweep → AUC=0.795 with ONLY 152 NEURONS
  |
  |        THE CENTRAL MYSTERY CRACKS OPEN:
  |        The AF signal lives in dense residual stream geometry.
  |        SAEs decompose it and destroy it. Linear probes preserve it.
  |        152 of 5,376 dimensions (2.8%) encode alignment faking intent.
  |
Jan 30 ─── Project Hindsight v2. This document.
```

---

## THE CENTRAL MYSTERY (partially solved)

```
                    SAME TASK
                    SAME DATA

  Fine-tuned         SAE Feature         Sparse Linear
  Detectors          Sweeps              Probe (L40)
  ┌─────────┐        ┌─────────┐        ┌─────────┐
  │  0.98   │        │  0.47   │        │  0.795  │
  │  AUROC  │        │  AUROC  │        │  AUROC  │
  └─────────┘        └─────────┘        └─────────┘
       │                  │                  │
       ▼                  ▼                  ▼
    WORKS             RANDOM             WORKS (sparse)

  WHY SAEs FAIL: The AF signal lives in the dense geometry
  of the residual stream — how 152 dimensions interact as
  a subspace. SAEs decompose representations into individual
  features and destroy this geometric structure. It's like
  analyzing a chord by looking at individual piano keys:
  C, E, and G separately are just notes. Together they're
  a C major chord. SAEs see the notes. The probe hears
  the chord.
```

**Original hypothesis (Jan 22):** AF is *per-trace* — each instance reasons
its way to deception uniquely. The only cross-trace constant is vocabulary.

**Updated hypothesis (Jan 30):** AF intent IS cross-trace, but it lives in
**dense subspace geometry**, not individual features. A sparse set of 152
residual stream dimensions collectively encode "this model is scheming" vs
"this model is being honest." The signal is relational (how dims interact),
not featural (which individual features fire). Fine-tuned detectors learn this
subspace implicitly. SAE probes can't see it because decomposition is the
wrong operation.

**Remaining gap:** The linear probe gets 0.795, fine-tuned gets 0.98. The
remaining 0.185 likely comes from non-linear interactions between the 152
dimensions and information from multiple layers. A two-layer MLP on the
selected dimensions or a cascade architecture may close this gap.

---

## THE CASCADE: HOW TO DEPLOY THIS

The sparse probe is a **validator**, not a detector. It ranks well (AUC=0.795)
but classifies poorly (FPR=0.93). The correct deployment is a two-stage
pipeline:

```
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Black-Box SFT Detector (AUROC ≈ 0.98)         │
│   Input: raw scratchpad text                            │
│   Output: "this might be alignment faking"              │
│   Role: DETECTION (fast, accurate, opaque)              │
└──────────────────────┬──────────────────────────────────┘
                       │ flagged candidates
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2: Sparse 152-Dim Probe (AUROC ≈ 0.795)          │
│   Input: layer 40 hidden states                         │
│   Output: "neurons 924, 2465, 4145 explain why"         │
│   Role: EXPLANATION (sparse, interpretable, auditable)  │
└──────────────────────┬──────────────────────────────────┘
                       │ evidence report
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 3: Human Review                                   │
│   Sees: detector score + per-neuron evidence            │
│   Can verify, override, or escalate                     │
└─────────────────────────────────────────────────────────┘
```

---

## REPOS & ARTIFACTS

```
 REPO                         COMMITS   DATES              STATUS    PURPOSE
 ───────────────────────      ───────   ──────────────     ───────   ─────────────────────
 mindreader                       87   Dec 29 - Jan 14    Frozen    Training framework
 lightbright                     122   Jan 3 - Jan 22     Public    SAE experiments
 af-detection-benchmark           30   Jan 6 - Jan 11     Public    Eval dataset
 BoxGlass                         88   Jan 3 - Jan 22     Private   GemmaScope experiments
 SAE_Cluster_Probe_Detector       20+  Jan 27 - Jan 30    Public    Systematic re-eval + Exp 11
 unsloth-mlx                      10   Jan 4 - Jan 6      Active    Mac MLX inference
 GlassBox                         --   (non-git)          Frozen    Early experiments

 HUGGINGFACE MODELS (5)
 ──────────────────────────────────────────────────
 vincentoh/gemma3-27b-af-detector-v2    0.980 AUROC
 vincentoh/gpt-oss-20b-af-detector     0.964 AUROC
 vincentoh/gpt-oss-120b-af-detector    0.956 AUROC
 vincentoh/gemma3-4b-af-detector       0.940 AUROC
 vincentoh/llama-8b-af-detector        0.690 AUROC

 HUGGINGFACE DATASETS (4)
 ──────────────────────────────────────────────────
 vincentoh/af-detection-benchmark      2,330 samples (5 classes)
 vincentoh/alignment-faking-training   incl. gold_106 (106 samples)
 vincentoh/af-model-organisms          707 samples (SAE probe-filtered)
 vincentoh/sycophant-af-samples        319 samples

 KEY LOCAL ARTIFACTS (experiments/11_mech_interp_intent/)
 ──────────────────────────────────────────────────
 saved_states/full_states_11c.npz      114 MB  Hidden states, 2,786 samples
 saved_states/best_probe_weights.npz   109 KB  Full 5,376-dim weight vector
 saved_states/sparse_l1_C0.045_best.npz  87 KB  The 152-dim probe
 results_11d.json                       10 KB  Best results + ablation + CV
```

---

## LESSONS & MISTAKES

### Things Done Right
1. **Caught own data leakage** -- 0.83 was wrong, published honest 0.47
2. **Created compliant_hard_negative class** -- proved vocabulary confound conclusively
3. **Cross-architecture validation** -- didn't rely on one model family
4. **Published negative results** -- SAE failure is informative
5. **gold_106 discipline** -- never trained on it, eval only, across all 32 days
6. **Didn't stop at the negative result** -- Phase 5 found the signal SAEs missed
7. **Kept the box up** -- ran 4 experiments in one H100 session, got the breakthrough

### Things That Cost Time
1. **Feature 136 ablation bug (Jan 1-2)** -- false 100% flip rate, methodology error
2. **Causal experiments v1-v5 (Jan 3)** -- all invalid due to broken classifier
3. **Original 0.83 AUROC claim** -- took days to discover the leakage
4. **6 repos for one project** -- fragmented history, hard to reconstruct
5. **DAG experiments (Exp 07-10b)** -- structural features fundamentally can't work
   because AF is semantic not structural. Should have pivoted to dense reps sooner.

### Open Questions
- Is the per-trace hypothesis dead, or does it explain the 0.795→0.98 gap?
- Can a 2-layer MLP on the 152 dims close the gap to 0.9+?
- Do the 152 dimensions transfer across models (Gemma → Llama → Claude)?
- What does dimension 924 (strongest aligned indicator) encode semantically?
- Would SAE decomposition of the 152-dim subspace recover interpretable features?
- Is the cascade (detector + probe) deployable in a real monitoring system?

---

## WHAT'S IN THE ZIP (for future deep dives)

```
 /Users/vincent/Desktop/Jan2026claude-projects-backup.zip  (237 MB)
 └── 17 Claude Code project histories
     ├── mindreader (main, data, gemma-scope-2, lambda-backup)
     ├── lightbright (main, docs, multimax, contrastive, adversarial, sae-detection)
     ├── af-detection-benchmark
     ├── BoxGlass (main, audit, gen-data, gen-data2, jan5-llama)
     └── unsloth-mlx
```

---

## ONE-LINE SUMMARY

> Built detectors that beat Anthropic's baseline (0.98 vs 0.92), discovered SAE features capture vocabulary not intent (the honest 0.47), then found the signal lives in 152 dimensions of dense residual stream geometry (0.795) — cracking open the central mystery of why fine-tuned models succeed where interpretability methods fail.

---

*Generated: January 30, 2026 | Project Hindsight v2*
*Updated with Phase 5 results from Experiment 11 mechanistic interpretability*
