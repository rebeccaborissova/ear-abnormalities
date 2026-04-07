# Infant Ear Landmark Confidence Scoring Implementation

This document describes the confidence scoring system implemented for the infant ear landmark detection model.

# Confidence Scoring — Infant Ear Landmark Detection
---

## Files

| File | Role |
|---|---|
| `infant_confidence_utils.py` | All confidence math, coloring, and drawing logic |
| `infant_infer_confidence_analysis.py` | Runs inference and routes images by confidence |

---

## How a Confidence Score is Computed

Each of the 23 landmarks has its own heatmap output — a 2D array where high values
indicate where the model thinks the landmark is. Two properties of that heatmap are
combined to produce a single confidence score per landmark.

### Step 1 — Heatmap peak (weight: 0.7)

```
peak = max value anywhere in the heatmap
```

A sharp, concentrated heatmap has a high peak. A flat or noisy heatmap has a low
peak. This is the dominant signal — 70% of the final score.

### Step 2 — Heatmap entropy (weight: 0.3)

```
entropy = measure of how spread out the heatmap probability mass is
```

The heatmap is normalized to sum to 1 (treated as a probability distribution), then
Shannon entropy is computed over it. Low entropy = probability is concentrated in one
place = confident. High entropy = probability is spread everywhere = uncertain.

Entropy is inverted and capped at a max of 10.0 before combining:

```
entropy_norm = 1.0 - min(entropy / 10.0, 1.0)
```

### Step 3 — Combined score

```
confidence = (0.7 × peak) + (0.3 × entropy_norm)
```

Result is clipped to [0.0, 1.0].

This is computed independently for all 23 landmarks, producing a confidence array
of shape `(23,)`.

### Step 4 — Total image confidence

The 23 per-landmark scores are aggregated into one number for the whole image:

| Metric | Meaning |
|---|---|
| `mean` (default) | Average confidence across all landmarks |
| `median` | Middle value — less sensitive to a few bad landmarks |
| `min` | Worst single landmark — strictest, fails if any landmark is uncertain |

---

## Confidence Bands



| Band | Total confidence | Dot color |
|---|---|---|
| HIGH | >= 0.8 | Green |
| MEDIUM | >= 0.6 and < 0.8 | Yellow |
| LOW | < 0.6 | Red |


---

## Inference Pipeline Flow

```
Image
  │
  ▼
Model forward pass
  │
  ▼
23 heatmaps  (shape: 23 × H × W)
  │
  ├── soft_argmax_2d ──► 23 (x, y) coordinates  ← the actual prediction
  │
  └── get_confidence_for_landmarks()
        │
        ├── per landmark: (0.7 × peak) + (0.3 × entropy_norm)
        │
        └── 23 confidence scores  (shape: 23,)
              │
              └── get_total_confidence(metric='mean')
                    │
                    └── 1 total score
                          │
                    ┌─────┴──────┐
                  >= 0.8       >= 0.6       < 0.6
                    │             │            │
              high_confidence/ medium_/    low_confidence/
              folder           confidence/  folder
                                folder
```

---

## Output Structure

```
inference_results/
├── high_confidence/       # total confidence >= 0.8
├── medium_confidence/     # total confidence >= 0.6 and < 0.8
├── low_confidence/        # total confidence < 0.6
├── statistics/
└── confidence_report.txt  # per-image scores and summary stats
```

---

## Running Inference

```bash
python infant_infer_confidence_analysis.py \
    --model  /path/to/model.pth \
    --images /path/to/images/ \
    --output inference_results \
    --metric mean
```

| Argument | Default | Options |
|---|---|---|
| `--model` | required | path to `.pth` checkpoint |
| `--images` | required | directory of `.jpg/.png/.bmp` images |
| `--output` | `inference_results` | any directory name |
| `--metric` | `mean` | `mean`, `median`, `min` |

