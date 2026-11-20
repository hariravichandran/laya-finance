# Laya-Finance reproduction status

Last updated 2026-11-19. Covers the published checkpoint `hravi/laya-finance` (third checkpoint, uploaded 2026-10-12) and the benchmark numbers in the root README.

## Verified

| Check | Result |
|---|---|
| Hugging Face `model.safetensors` vs local J checkpoint (`runs/mt_j/final`) | Byte-identical, SHA-256 `ec1a020c8e575bc8…baff` |
| README Laya-Finance rows vs `results/laya_mt_j.json` | All eight match |
| Original Laya row vs `results/laya_zeroshot.json` | Match |
| ProsusAI/finbert row vs `results/finbert.json` | Match |
| gpt-oss-20b rows | Rerun on full test sets, 2026-11-16 (`results/llm_gptoss20b_full.json`, `results/llm_gptoss20b_new.json`); 0 unparsed answers |
| Sentiment and topic train sets | Deduplicated against test sets by normalized text (`build_dataset.py`, `build_multitask.py`) |
| `laya-router` | Private GitHub repo. No Laya-Router model on Hugging Face |

## Known differences between runs

- gpt-oss-20b on the same items moved by up to about one point between runs (FiQA 0.824 then 0.834). The 250-item sample run is in `results/superseded/` and is not cited.
- The two Laya-Finance J result files disagree on timing. `laya_mt_j.json` gives 10.7 ms per item for FOMC; `laya_mt_j_items.json` (not in this repo) gives 302 ms for the same task, with no log or recorded device. The README timing range comes from `laya_mt_j.json` only.

## Still open

1. **Timing source.** Resolved 2026-11-19 with a controlled benchmark (`results/latency/`): median of three repeats on an AMD Radeon 8060S, torch 2.14.1+rocm7.2, laya 0.4.34. GPU: 3–28 ms per item at batch 32, 24–33 ms at batch 1. CPU: 41–395 ms per item at batch 32. GPU idle state recorded: 0% busy for the five seconds before the run, about 93% mean during it (`results/latency/gpu_busy_samples_gpu_512.txt`). A rerun gave the same medians within 1%.
2. **Branch drift.** Resolved 2026-11-19: the second-checkpoint branch `finance/finetune-benchmarks` (`e9aa89f`) was deleted from GitHub. It had no commits that were not already on `main`.
3. **Hugging Face model card.** Updated 2026-11-19: `209a128` added the full-test-set gpt-oss figures, the timing device and the PhraseBank note; `29f4969` replaced the timing range with the controlled benchmark. Weights unchanged.
4. **Raw training data.** Not in this repo and not required for the metrics above. It stays in the private corpus repository.

## Result files

| File | Contents |
|---|---|
| `results/laya_mt_j.json` | Laya-Finance, all tasks, with timing |
| `results/laya_zeroshot.json` | Original Laya, zero-shot |
| `results/finbert.json` | ProsusAI/finbert, sentiment |
| `results/llm_gptoss20b_full.json` | gpt-oss-20b, full test sets for sentiment, topic, FOMC and headline |
| `results/llm_gptoss20b_new.json` | gpt-oss-20b, full test sets for terminology and trading style |
| `results/logs/` | Command and log for the full gpt-oss-20b run |
| `results/superseded/llm_gptoss20b_250sample.json` | Earlier gpt-oss-20b run on 250-item samples; superseded |

The result files contain metrics only (accuracy, macro-F1, calibration, timing, sample counts). They contain no test text.
