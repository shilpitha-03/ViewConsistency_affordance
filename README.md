# Diagnosing View-Consistency Failures in Affordance Prediction Models

A systematic diagnostic study of **view-consistency in 2D vision-language affordance models** —
testing the assumption that part-level understanding in modern vision encoders yields
viewpoint-stable affordance predictions, and finding that it largely does, *as long as the
functional part stays visible*.

## TL;DR

- Affordance-based manipulation systems assume predictions stay reliable under viewpoint change.
  This is the first systematic test of that assumption for 2D vision-language affordance models.
- The dominant failure axis is **not viewpoint — it's functional-part visibility.** When a mug's
  handle is visible, grasp prediction is saturated and view-invariant (mean `grasp_max` = **0.992**);
  under full occlusion it collapses to **0.562**.
- The model-internal **`grasp_max` similarity score** is a **training-free, per-image confidence
  signal** a downstream Vision-Language-Action (VLA) policy can read to trigger active perception or
  occlusion-aware behavior — no retraining of the affordance backbone required.

## Background

Robots rarely see the canonical front-facing views that dominate affordance benchmarks — a mobile
robot approaches a mug from the side, a manipulation arm sees objects from oblique angles. If
affordance predictions degrade under viewpoint variation, every downstream system built on them
silently inherits that failure. This project diagnoses *whether and why* that degradation happens.

## Approach

### Two-mode failure taxonomy (starting point)

- **Identity failure** — the model fails to recognize the object from a new viewpoint, so any
  affordance prediction is unreliable by construction.
- **Affordance failure** — the model recognizes the object but localizes the wrong interaction region.

(The results show this taxonomy is too coarse — see [Findings](#key-findings).)

### Models evaluated

| Model | Backbone | Output | Verdict as a diagnostic target |
|---|---|---|---|
| **RoboPoint** | CLIP | sparse (x, y) keypoints, scene-relative | **Unsuitable** — predicts scene-relative points, not functional parts; CLIP gives weak part-level features |
| **GAT** (Aff-Grasp) | DINOv2 | dense, part-grounded affordance heatmaps | **Primary target** — dense per-pixel output is measurable across views |

### GAT architecture & the `grasp_max` signal

GAT = DINOv2 ViT-B/14 (LoRA-adapted) + a Depth Feature Injector (DFI) that cross-attends
pseudo-depth (from Depth-Anything) into the patch stream, producing dense heatmaps over 8
affordance categories (grasp, cut, scoop, pound, support, screw, contain, stick) via cosine
similarity to learned category embeddings.

`grasp_max(I) = max over pixels of the normalized grasp-category cosine similarity` — the model's
peak confidence that *any* visible region is graspable. It needs no per-pixel ground truth, is
readable by any downstream policy, and is interpretable as a confidence score.

### DFI ablation (3 inference-time conditions)

`no DFI` (pure DINOv2 + LoRA) · `zero depth` (DFI active, null signal) · `real depth`
(Depth-Anything pseudo-depth). Because the DFI was present during training, geometric priors are
baked into the weights even when bypassed — so this isolates the *marginal* inference-time effect
of explicit depth.

## Experimental setup

Three phases, progressively removing confounds:

1. **RoboPoint on synthetic ShapeNet renders** (mugs/chairs, 5 azimuths).
2. **GAT on the same synthetic renders** — characterizes the synthetic domain gap.
3. **GAT on real photographs** (mug, knife, scissors, bowl) — the primary analysis, with explicit
   handle-visible vs handle-hidden strata.

## Key findings

**RoboPoint is not a viable diagnostic target.** Three confounds can't be jointly controlled:
prompt/model mismatch (it grounds affordance in scene context, not functional parts),
synthetic-render domain gap, and a sparse output too coarse for per-pixel view consistency.

**GAT on synthetic data is dominated by domain gap** — activation tracks specular highlights and
background contrast rather than object geometry, so the analysis moves to real photographs.

**Core result — failure is gated on functional-part visibility, not viewpoint:**

| Condition | `grasp_max` |
|---|---|
| Mug handle visible (floral, 5 views) | 0.988–0.995 (mean **0.992**) |
| Mug handle fully occluded (red mug, az2) | **0.562** |
| Knife — grasp (handle) across 5 views | 0.990–0.994 |
| Knife — cut (blade) across 5 views | 1.000 |
| Bowl — contain (all views) | 1.000 |
| Bowl — grasp (all views) | 0.367–0.508 (**fails**) |

- The handle-hidden *mean* (0.902) understates the effect — only az2 is fully occluded; partially
  visible handles recover above 0.97. The failure is gated on **handle pixels being in view**, not
  on the verbal viewing angle.
- **Knife = positive control:** two spatially distinct affordances stay stable across views.
- **Bowl = controlled negative:** `contain` succeeds but `grasp` fails *in fully visible, canonical
  views* — isolating a second failure source, **dataset bias** (egocentric training has no
  bowl-grasp interactions), distinct from occlusion.

**Depth provides marginal recovery, not rescue.** Under strict occlusion: `no DFI` 0.562,
`zero depth` 0.490, `real depth` 0.604 — all below the 0.8 reliability threshold. Depth sees the
visible silhouette, not the occluded handle, so it can't reconstruct the missing affordance. The
bottleneck is missing training annotations, not geometric perception.

**The two-mode taxonomy is too coarse.** A third axis dominates — *functional-part visibility* —
motivating a finer taxonomy separating (a) object identity, (b) functional-part visibility, and
(c) category-level dataset coverage.

## Proposed application: a training-free confidence gate

Insert `grasp_max` as a gate in the Aff-Grasp pipeline between GAT and Contact-GraspNet:

```
grasp_max ≥ τ  → PROCEED to Contact-GraspNet
grasp_max < τ  → HOLD, move camera (active perception)
```

with τ = 0.8 (cleanly separates handle-visible ≥ 0.988 from strict-occlusion ≤ 0.604). On a
cluttered 7-view table scene every view scored ≥ 0.919 (no false negatives); mask-fragmentation
between best (3 connected regions) and worst (24 regions) views shows why score variation is
meaningful even above threshold. Zero additional training — it reads GAT's existing output.

## Limitations

- `grasp_max` is a max-pool — it discards spatial structure (a tight handle activation and a
  diffuse one can score the same); pair it with spatial entropy / connected-component count.
- It conflates **occlusion** (recoverable by moving the camera) with **dataset absence**
  (not recoverable) — both produce the same low score.
- Small dataset; only one fully-occluded view; single instance per category (1 knife, 1 bowl,
  2 mugs).

## Future work

- Pair `grasp_max` with a spatial-coherence metric for a better-calibrated confidence signal.
- Validate the confidence gate end-to-end in Aff-Grasp with Contact-GraspNet.
- Broader object coverage (bottles, cans, handleless cups) to test whether the dataset-bias failure
  generalizes beyond bowls.

## Repository structure
<!-- TODO: list main folders/files, e.g. robopoint/, gat/, data/, ablation/, notebooks/ -->

## Setup
<!-- TODO: dependencies + install; note GAT / RoboPoint / Depth-Anything checkpoints needed -->

## Usage / reproducing
<!-- TODO: how to run each phase (RoboPoint renders, GAT real-photo eval, DFI ablation) -->
