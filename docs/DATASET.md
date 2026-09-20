# Dataset protocol

The committed dataset contains 50,000 deterministic examples: 35,000 train, 5,000 dev, 5,000 IID test and 5,000 held-out wording/style examples.

Controls are exactly 37% of the dataset. Supported examples are exactly 63% and are balanced across 12 operations (2,625 examples per operation across all splits).

Positive lexical sources are exactly 30% real words, 50% algorithmically augmented real words and 20% random strings. Non-held-out task prompts combine the native varied generator with reconstructed external-benchmark-style wording. Held-out prompts use template/wrapper families absent from train/dev/IID test.

Ground truth is produced only by deterministic code. `scripts/generate_dataset.py` reproduces the committed JSONL files from the bundled lexical seed list and fixed seed.
