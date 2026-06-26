# Diagnosing View-Consistency Failures in Affordance Prediction Models

A diagnostic study of *why* affordance-prediction models fail across viewpoints — and a
training-free signal a downstream policy can use to react to those failures.

## Overview

Affordance models are assumed to fail mainly when the **viewpoint** changes. We test that
assumption across two pipelines and find the dominant failure axis is actually whether the
**functional part is visible**, not the camera angle.

## Pipelines studied

- **RoboPoint / CLIP**
- **GAT / DINOv2**

## Key findings

- The dominant failure axis is **functional-part visibility**, not viewpoint.
- Grasp confidence falls from **0.99 (part visible)** to **0.56 (fully occluded)**.
- The model's internal **grasp-confidence score** works as a **training-free signal**: a
  downstream VLA policy can read it to trigger **active perception** (move to get a better
  view) — **no retraining** of the affordance backbone required.

## Repository structure
<!-- TODO: list main folders/files -->

## Setup
<!-- TODO: dependencies + install -->

## Usage / reproducing the analysis
<!-- TODO: how to run the experiments / regenerate the figures -->

## Status
Graduate course research project, Spring 2026.
