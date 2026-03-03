# UI Semantic Boxer README

`ui_semantic_boxer.py` is a utility for generating single-step clickable UI data from a live device screen.

It combines:
- hierarchy (for clickable boxes)
- VLM (for semantic text / ui_kind / ambiguity handling)

and exports:
- `ui_semantics.json`
- per-item `actions.json` / `react.json`
- debug JSON files for VLM decisions

## 1. Quick Start

Use the wrapper script:

```bash
bash ./run-ui-semantic-boxer.sh
```

Main config file:
- [`run-ui-semantic-boxer.sh`](/home/zhaoxi/ipads/llm-agent/test/MobiAgent/run-ui-semantic-boxer.sh)

Core implementation:
- [`collect/auto/ui_semantic_boxer.py`](/home/zhaoxi/ipads/llm-agent/test/MobiAgent/collect/auto/ui_semantic_boxer.py)

## 2. Pipeline

1. Capture screenshot + hierarchy from Android/Harmony.
2. Extract clickable nodes and bboxes.
3. Build `text`:
- default: use hierarchy text when available, VLM for missing text
- `vlm_text_only=on`: use VLM text for all clickable nodes
4. Build `ui_kind`:
- rule first
- optional page-level VLM review
5. Refine text / phrase ambiguity (optional VLM stages).
6. Export semantic index and per-item action data.

## 3. Important Parameters

From `run-ui-semantic-boxer.sh`:

- `USE_VLM`: enable/disable VLM usage.
- `VLM_TEXT_ONLY`: if `on`, ignore hierarchy text and caption every clickable node with VLM.
- `ENABLE_KIND_VLM`: page-level `ui_kind` verification.
- `ENABLE_VLM_TEXT_REFINE`: refine generic VLM labels (e.g. `状态标识`, `空白区域`).
- `ENABLE_DUPLICATE_DESC_VLM`: disambiguate same-label targets at phrase level.
- `ENABLE_TASK_DESC_VLM_REVIEW`: final page-level check for task description ambiguity.

## 4. Output Structure

Typical output directory:

`collect/auto/ui-semantic-output/<timestamp>/`

Contains:
- `screenshot.jpg`
- `annotated.jpg`
- `hierarchy.xml` or `hierarchy.json`
- `ui_semantics.json`
- `<id>/actions.json`
- `<id>/react.json`
- `<id>/1.jpg`
- `<id>/1_annotated.jpg`

## 5. Debug Files

To inspect VLM behavior:
- `ui_kind_vlm_review_debug.json`
- `vlm_text_refine_debug.json`
- `task_description_ambiguity_review_debug.json`

These files record request context, parsed outputs, and merge/rewrite decisions.

## 6. Notes

- `OPENROUTER_API_KEY` is required when `USE_VLM=on`.
- `vlm_text_only=on` requires `USE_VLM=on`.
- Larger `MAX_ITEMS` increases VLM cost and latency.
