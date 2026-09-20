# Configuration reference

Experiment defaults live in TOML files under `configs/`.

## Cache

```toml
[cache]
auto_download = true
cache_dir = ""
revision = "..."
```

CLI `--auto-download` / `--no-auto-download` and `--model-cache-dir` override these values where exposed. Local snapshots are always preferred over network downloads.

## Recovery

```toml
[recovery]
heartbeat_seconds = 15
checkpoint_every_examples = 50
```

Training state stores active process time separately from wall-clock downtime.

## Evaluation

The primary tool-policy config defaults to both IID and held-out evaluation and resumable progress. The complete pipeline inherits these defaults and can run with no CLI arguments after training.

## Result injection

`architecture.candidate_layers` defines every injection layer trained and analyzed. The selected deployment layer is chosen on dev only; test and held-out splits are measurement-only.
