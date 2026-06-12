# FDS Sentiment Module

The paper describes FDS as DeepSeek-V4 with LoRA supervised fine-tuning followed by GRPO reinforcement learning. The PDF does not include model weights, prompts, or the final training corpus, so this repository keeps the interface reproducible and replaceable:

- `news.csv` can include a ready-made `sentiment` column.
- If only `text` is present, preprocessing uses a lightweight lexicon fallback.
- `hd_tilag.features.fds_adapter.TransformersFDSAdapter` can annotate text with any Hugging Face text-classification checkpoint whose labels map to positive/neutral/negative.

For a paper-faithful run, train or obtain an FDS-compatible classifier, annotate each benchmark's `news.csv`, then run preprocessing. The HD-TILAG model consumes only the daily positive/neutral/negative proportions defined in Eq. (5), so the downstream architecture is unchanged.
