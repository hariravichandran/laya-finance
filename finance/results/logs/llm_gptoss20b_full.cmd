# Full-test-set gpt-oss:20b zero-shot baseline (no sampling; --max-per-file set above any file size)
# Started: $(date) ; interpreter: ~/.venvs/laya-eval (numpy 2.5.5, scikit-learn 1.9.1)
~/.venvs/laya-eval/bin/python eval_llm_baseline.py --model gpt-oss:20b --tag llm_gptoss20b_full --tasks sentiment,topic,fomc,headline --max-per-file 100000 --workers 2 --seed 1
2026-11-16 05:03:19
