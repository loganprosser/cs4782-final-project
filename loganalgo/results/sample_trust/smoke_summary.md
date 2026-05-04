# Sample Trust Run Summary

Generated: 2026-05-04 02:53:45

Run filter: `sanity_`
Runs summarized: 8

Outputs:

- Summary CSV: `results/sample_trust/summary.csv`
- Per-run metrics and plots: `results/sample_trust/<run_name>`
- Aggregate plots: `results/sample_trust/aggregate_plots`

## Top Runs by Best Test Accuracy

| rank | run_name | model | mode | batch_size | seed | best_test_acc | final_test_acc | final_trust_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | sanity_none | mlp | none | 32 | 0 | 0.3633 | 0.3633 | 1.0000 |
| 2 | sanity_entropy | mlp | entropy | 32 | 0 | 0.3633 | 0.3633 | 0.1003 |
| 3 | sanity_combined_simple | mlp | combined_simple | 32 | 0 | 0.3633 | 0.3633 | 0.0500 |
| 4 | sanity_confidence | mlp | confidence | 32 | 0 | 0.3164 | 0.3164 | 0.1040 |
| 5 | sanity_inverse_loss | mlp | inverse_loss | 32 | 0 | 0.3164 | 0.3164 | 0.1040 |
| 6 | sanity_final_layer_align_mag | mlp | final_layer_align_mag | 32 | 0 | 0.2383 | 0.2383 | 0.1294 |
| 7 | sanity_final_layer_align | mlp | final_layer_align | 32 | 0 | 0.1758 | 0.1758 | 0.2385 |
| 8 | sanity_ema_final_layer_align | mlp | ema_final_layer_align | 32 | 0 | 0.1133 | 0.1133 | 0.1308 |

## By Model and Trust Mode

| model | mode | runs | mean_best_test_acc | best_test_acc | best_batch_size | best_run |
| --- | --- | --- | --- | --- | --- | --- |
| mlp | combined_simple | 1 | 0.3633 | 0.3633 | 32 | sanity_combined_simple |
| mlp | confidence | 1 | 0.3164 | 0.3164 | 32 | sanity_confidence |
| mlp | ema_final_layer_align | 1 | 0.1133 | 0.1133 | 32 | sanity_ema_final_layer_align |
| mlp | entropy | 1 | 0.3633 | 0.3633 | 32 | sanity_entropy |
| mlp | final_layer_align | 1 | 0.1758 | 0.1758 | 32 | sanity_final_layer_align |
| mlp | final_layer_align_mag | 1 | 0.2383 | 0.2383 | 32 | sanity_final_layer_align_mag |
| mlp | inverse_loss | 1 | 0.3164 | 0.3164 | 32 | sanity_inverse_loss |
| mlp | none | 1 | 0.3633 | 0.3633 | 32 | sanity_none |

## By Batch Size and Trust Mode

| batch_size | mode | runs | mean_best_test_acc | mean_final_test_acc | mean_final_trust_mean |
| --- | --- | --- | --- | --- | --- |
| 32 | combined_simple | 1 | 0.3633 | 0.3633 | 0.0500 |
| 32 | confidence | 1 | 0.3164 | 0.3164 | 0.1040 |
| 32 | ema_final_layer_align | 1 | 0.1133 | 0.1133 | 0.1308 |
| 32 | entropy | 1 | 0.3633 | 0.3633 | 0.1003 |
| 32 | final_layer_align | 1 | 0.1758 | 0.1758 | 0.2385 |
| 32 | final_layer_align_mag | 1 | 0.2383 | 0.2383 | 0.1294 |
| 32 | inverse_loss | 1 | 0.3164 | 0.3164 | 0.1040 |
| 32 | none | 1 | 0.3633 | 0.3633 | 1.0000 |

## Delta vs Matched `none` Baseline

| model | batch_size | seed | mode | best_test_acc | baseline_best_test_acc | delta |
| --- | --- | --- | --- | --- | --- | --- |
| mlp | 32 | 0 | entropy | 0.3633 | 0.3633 | +0.0000 |
| mlp | 32 | 0 | combined_simple | 0.3633 | 0.3633 | +0.0000 |
| mlp | 32 | 0 | confidence | 0.3164 | 0.3633 | -0.0469 |
| mlp | 32 | 0 | inverse_loss | 0.3164 | 0.3633 | -0.0469 |
| mlp | 32 | 0 | final_layer_align_mag | 0.2383 | 0.3633 | -0.1250 |
| mlp | 32 | 0 | final_layer_align | 0.1758 | 0.3633 | -0.1875 |
| mlp | 32 | 0 | ema_final_layer_align | 0.1133 | 0.3633 | -0.2500 |

Interpretation note: this report is descriptive. Treat any improvement as a hypothesis until it survives repeated seeds and a full-data run.
