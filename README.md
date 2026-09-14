# Character Unit LLM — Oracle Injection Milestone

Current base model: `Qwen/Qwen3-4B`, loaded in 4-bit NF4 for the 8 GB RTX 4070 Laptop GPU.

This milestone contains two distinct components:

1. A deterministic character executor implementing `COUNT_CHAR`, `STRING_LENGTH`, `CHAR_AT`, `REVERSE`, and `FIND_CHAR`.
2. An **oracle residual-injection diagnostic** for `COUNT_CHAR`.

The oracle diagnostic deliberately supplies the correct operation and operands. It executes the count deterministically, converts the exact result into a representation using the frozen model's own token embedding space, injects that representation into one intermediate Qwen residual stream, and measures whether the probability/rank of the exact result changes.

This is **not** the final architecture and does **not** establish end-to-end accuracy. It only tests whether a deterministic result can enter and influence the frozen LM internally.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Baseline model probe

```powershell
python scripts/probe_model.py
```

## Oracle internal-injection diagnostic

```powershell
python scripts/oracle_injection.py
```

Default diagnostic:

```text
Question: How many "r" characters are in "strawberry"?
Oracle operation: COUNT_CHAR
Oracle operands: text="strawberry", character="r"
Deterministic result: 3
```

The script sweeps decoder layers `8,17,26,35` and residual strengths `0.25,0.5,1,2,4,8`. It records the next-token probability and rank of the exact deterministic result for every setting, then generates once with the strongest diagnostic setting.

Do not interpret the layer/scale sweep as a fair benchmark: it is intentionally an oracle interface test on the same example.
