# Off-the-shelf tool-calling comparisons

These experiments test whether the released model's existing function-calling
ability plus ordinary text feedback performs as well as the trained selector and
result mapper. They do not train an off-the-shelf model or inject hidden states.
The application still supplies the twelve character functions; no model includes
their Python executor inside its weights.

## Comparisons

1. **Released model + native tools + text feedback.** Qwen3-4B, Qwen3-8B,
   Phi-4-mini-instruct, and Hermes-3-Llama-3.1-8B use their supplied tokenizer chat
   templates and native function-call formats. Qwen thinking is disabled, as in
   the existing trained pipeline.
2. **Trained LoRA selector + text feedback.** The existing Phase-1 adapter makes
   the custom structured call. The executor's actual result returns through the
   backbone's native tool-response format. The adapter is disabled for answer
   generation, matching the mapper pipeline. No mapper is loaded.
3. **Trained LoRA selector + mapper.** Run the existing `evaluate_pipeline.py`.
   The architecture and training scripts are unchanged.

The second comparison helps isolate the mapper's contribution. It changes the
result interface and therefore also the answer-stage prompt/context; it is not a
claim that only one numerical tensor differs between conditions.

## Matched protocol

- Same frozen test and held-out files, prompts, twelve operations and executor.
- The executor runs only the model's predicted operation/arguments. Ground-truth
  labels are used only for scoring, never to repair a call or supply a result.
- Default native tool names are descriptive. `--mask-probability 0.5` reproduces
  the trained pipeline's randomized/masked catalog names, descriptions and order.
  Report both conditions separately. The LoRA comparison defaults to its existing
  configured mask probability (0.5).
- A maximum of one requested operation and one answer-generation stage. No
  retries, generated scripts, oracle calls, or forced ground-truth tool choices.
- Greedy generation, batch size one, the existing NF4 four-bit loading setup,
  192 call-generation tokens and 128 answer-generation tokens. The LoRA selector
  uses its own configured call budget. The full dataset is the primary comparison;
  sampled runs are pilots, not replacements for it.
- Sampled positive IDs match the unaided local evaluator's seed and stratified
  sampling. Controls are sampled separately by category. Full runs include all
  3,150 supported tasks and 1,850 controls in each evaluation split.
- Native numeric argument types are recorded separately. Integer strings are
  executable, as in the existing structured-call parser; floats and booleans are
  rejected. Literal string arguments are never coerced from lists or objects.
- All four model revisions are pinned. Original Qwen snapshots match the primary
  experiment pins. Reports also preserve the resolved snapshot path and hashes.

The native baseline is implemented with the released Hugging Face chat template,
not a custom `CALL` prompt and not the separate Qwen-Agent service framework.
Validate template compatibility before running a new snapshot. Runtime and
accuracy claims describe this specified inference implementation.

## Commands

Validate locally cached templates and dataset selection without loading weights:

```powershell
python scripts/run_tool_feedback_matrix.py --preflight
```

Pilot (two supported examples per operation and one control per category):

```powershell
python scripts/run_tool_feedback_matrix.py --examples-per-operation 2 --controls-per-category 1
```

Run all four off-the-shelf models on the full two evaluation splits:

```powershell
python scripts/run_tool_feedback_matrix.py
```

Models are reused locally. To download missing models, explicitly enable it and
choose a cache with enough space:

```powershell
python scripts/run_tool_feedback_matrix.py --auto-download --model-cache-dir D:/hf-models
```

Run Qwen native tools under the same name-masking condition as Phase 1:

```powershell
python scripts/run_tool_feedback_matrix.py --models qwen3_4b qwen3_8b --mask-probability 0.5
```

Run the trained-selector/text-feedback comparison after Phase-1 training:

```powershell
python scripts/evaluate_tool_feedback.py --config configs/baselines/tools/qwen3_4b.toml --policy-config configs/experiments/qwen3_4b/tool_policy.toml
```

`--with-lora-text` adds these trained-selector runs for selected Qwen models to
the matrix. Model generation requires CUDA; template preflight does not. Missing
models/adapters or unavailable CUDA are reported as failures, not zero accuracies.

## Scores and comparison

Every report retains the model output, predicted call, executed result, final
answer, and per-stage token/time measurements for each example. It reports:

- correct operation plus exact arguments;
- supported-task final-answer accuracy after text feedback;
- supported-task accuracy if the executor result were returned directly;
- scalar answer accuracy conditional on a correct call;
- control no-call accuracy, tool-attempt rate, and invalid-response rate;
- the existing pipeline's routing rule applied to the native predictions:
  scalar text feedback and direct return for string transforms.

Native text feedback is run for all executed operations. The matched-routing
accuracy/time/token fields are derived from those stages; they are not a separate
independently timed deployment. A native model can answer without a call; this
counts in full native final-answer accuracy, while the matched-routing score
requires an executed call, as the current trained pipeline does.

Top-level `exact_match_accuracy` and `example_count` refer to supported tasks,
matching the unaided baseline exports. `overall_system_success` also includes
controls; `evaluated_example_count` is the total number of tasks and controls.
Compare task-only scores with `supported_task_system_exact` in the mapper report.
Use `runtime_supported_tasks` when comparing against task-only unaided timing.

For paired supported-task comparisons, supply the actual report filenames:

```powershell
python scripts/compare_tool_feedback.py --pipeline results/qwen3_4b/tool_policy/pipeline_results.json --feedback <native-report.json> <lora-text-report.json>
```

The CSV aligns source example IDs separately for each split, checks their labels,
and reports accuracies on shared tasks and how many tasks only one system gets
right. It states whether task sets are identical; a pilot/full comparison must not
be presented as a full-set result. Existing metric/runtime exporters also discover
the native reports, but paired comparison avoids mixing control denominators.

Progress/report filenames include a protocol fingerprint covering selected IDs,
data hashes, native templates, model identity, generation settings and adapter
contents. Re-running the same command resumes completed examples; changed
conditions cannot silently reuse those answers.

## Provider documentation

- [Qwen3 function calling](https://github.com/QwenLM/Qwen3/blob/main/docs/source/framework/function_call.md)
- [Phi-4-mini model card](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [Microsoft function-calling example](https://github.com/microsoft/PhiCookBook/blob/main/md/02.Application/07.FunctionCalling/Phi4/FunctionCallingBasic/README.md)
- [Hermes-3 model card](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)
