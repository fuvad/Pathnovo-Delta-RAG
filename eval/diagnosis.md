# Diagnosis & Fix Plan

## Problem
Precision is 0.09 — we find the real changes but drown them in false positives.

## Root Causes

### 1. Too many noise elements (375 in Doc A, 496 in Doc B)
- 61 elements with <= 3 chars: "M", "26", "/", "*", ".", "RS", "E\nP"
- 31 single-character elements
- Grid labels (1-12, A-J) extracted as separate elements
- Nozzle IDs (N3207, N3208...) all become separate elements
- Block valve tags (43BL9008, 26BL9031...) all separate

### 2. Misclassification
- "26-KA-902" classified as PIPE (not EQUIPMENT)
- Short codes like "26" classified as TEXT
- Nozzle numbers and block valve tags are noise, not meaningful elements

### 3. Match threshold too low (0.4)
- Short similar strings match too easily
- "N3207" matches "N4207" with high semantic + type similarity
- Creates hundreds of false "modified" entries

## Fix Strategy

### A. Filter noise in ingestion (pdf_native.py)
- Skip elements <= 2 chars
- Skip pure number elements (grid labels)
- Skip single-letter grid labels (A, B, C...)

### B. Improve classifier (classifier.py)
- Fix equipment regex to catch "XX-KA-NNN" patterns
- Filter nozzle IDs as a known low-value type

### C. Raise match threshold (align.py)
- Increase MATCH_THRESHOLD from 0.4 to 0.55

### D. Improve ground truth matching (metrics.py)
- The fuzzy matcher may need adjustment too
