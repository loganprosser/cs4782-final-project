# Sample Trust Run Summary

Generated: 2026-05-04 03:23:30

Run filter: `sample_trust_20260504_025508_`
Runs summarized: 64

Outputs:

- Summary CSV: `results/sample_trust/summary.csv`
- Per-run metrics and plots: `results/sample_trust/<run_name>`
- Aggregate plots: `results/sample_trust/aggregate_plots`

## Top Runs by Best Test Accuracy

| rank | run_name | model | mode | batch_size | seed | best_test_acc | final_test_acc | final_trust_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | sample_trust_20260504_025508_cnn_final_layer_align_bs32_lr0p001_seed0 | cnn | final_layer_align | 32 | 0 | 0.9875 | 0.9875 | 0.1500 |
| 2 | sample_trust_20260504_025508_cnn_entropy_bs32_lr0p001_seed0 | cnn | entropy | 32 | 0 | 0.9850 | 0.9850 | 0.9818 |
| 3 | sample_trust_20260504_025508_cnn_final_layer_align_bs16_lr0p001_seed0 | cnn | final_layer_align | 16 | 0 | 0.9845 | 0.9845 | 0.1676 |
| 4 | sample_trust_20260504_025508_cnn_entropy_bs16_lr0p001_seed0 | cnn | entropy | 16 | 0 | 0.9830 | 0.9795 | 0.9861 |
| 5 | sample_trust_20260504_025508_cnn_final_layer_align_mag_bs16_lr0p001_seed0 | cnn | final_layer_align_mag | 16 | 0 | 0.9830 | 0.9770 | 0.1493 |
| 6 | sample_trust_20260504_025508_cnn_combined_simple_bs16_lr0p001_seed0 | cnn | combined_simple | 16 | 0 | 0.9830 | 0.9830 | 0.1144 |
| 7 | sample_trust_20260504_025508_cnn_none_bs32_lr0p001_seed0 | cnn | none | 32 | 0 | 0.9830 | 0.9645 | 1.0000 |
| 8 | sample_trust_20260504_025508_cnn_inverse_loss_bs16_lr0p001_seed0 | cnn | inverse_loss | 16 | 0 | 0.9825 | 0.9825 | 0.9898 |
| 9 | sample_trust_20260504_025508_cnn_ema_final_layer_align_bs32_lr0p001_seed0 | cnn | ema_final_layer_align | 32 | 0 | 0.9825 | 0.9825 | 0.1243 |
| 10 | sample_trust_20260504_025508_cnn_final_layer_align_mag_bs128_lr0p001_seed0 | cnn | final_layer_align_mag | 128 | 0 | 0.9820 | 0.9820 | 0.1174 |
| 11 | sample_trust_20260504_025508_cnn_confidence_bs16_lr0p001_seed0 | cnn | confidence | 16 | 0 | 0.9815 | 0.9790 | 0.9886 |
| 12 | sample_trust_20260504_025508_cnn_none_bs128_lr0p001_seed0 | cnn | none | 128 | 0 | 0.9815 | 0.9815 | 1.0000 |
| 13 | sample_trust_20260504_025508_cnn_none_bs16_lr0p001_seed0 | cnn | none | 16 | 0 | 0.9810 | 0.9785 | 1.0000 |
| 14 | sample_trust_20260504_025508_cnn_ema_final_layer_align_bs16_lr0p001_seed0 | cnn | ema_final_layer_align | 16 | 0 | 0.9810 | 0.9735 | 0.1142 |
| 15 | sample_trust_20260504_025508_cnn_final_layer_align_mag_bs32_lr0p001_seed0 | cnn | final_layer_align_mag | 32 | 0 | 0.9810 | 0.9780 | 0.1216 |

## By Model and Trust Mode

| model | mode | runs | mean_best_test_acc | best_test_acc | best_batch_size | best_run |
| --- | --- | --- | --- | --- | --- | --- |
| cnn | combined_simple | 4 | 0.9808 | 0.9830 | 16 | sample_trust_20260504_025508_cnn_combined_simple_bs16_lr0p001_seed0 |
| cnn | confidence | 4 | 0.9795 | 0.9815 | 16 | sample_trust_20260504_025508_cnn_confidence_bs16_lr0p001_seed0 |
| cnn | ema_final_layer_align | 4 | 0.9791 | 0.9825 | 32 | sample_trust_20260504_025508_cnn_ema_final_layer_align_bs32_lr0p001_seed0 |
| cnn | entropy | 4 | 0.9820 | 0.9850 | 32 | sample_trust_20260504_025508_cnn_entropy_bs32_lr0p001_seed0 |
| cnn | final_layer_align | 4 | 0.9830 | 0.9875 | 32 | sample_trust_20260504_025508_cnn_final_layer_align_bs32_lr0p001_seed0 |
| cnn | final_layer_align_mag | 4 | 0.9818 | 0.9830 | 16 | sample_trust_20260504_025508_cnn_final_layer_align_mag_bs16_lr0p001_seed0 |
| cnn | inverse_loss | 4 | 0.9803 | 0.9825 | 16 | sample_trust_20260504_025508_cnn_inverse_loss_bs16_lr0p001_seed0 |
| cnn | none | 4 | 0.9812 | 0.9830 | 32 | sample_trust_20260504_025508_cnn_none_bs32_lr0p001_seed0 |
| mlp | combined_simple | 4 | 0.9571 | 0.9605 | 16 | sample_trust_20260504_025508_mlp_combined_simple_bs16_lr0p001_seed0 |
| mlp | confidence | 4 | 0.9547 | 0.9580 | 16 | sample_trust_20260504_025508_mlp_confidence_bs16_lr0p001_seed0 |
| mlp | ema_final_layer_align | 4 | 0.9526 | 0.9625 | 16 | sample_trust_20260504_025508_mlp_ema_final_layer_align_bs16_lr0p001_seed0 |
| mlp | entropy | 4 | 0.9604 | 0.9615 | 64 | sample_trust_20260504_025508_mlp_entropy_bs64_lr0p001_seed0 |
| mlp | final_layer_align | 4 | 0.9597 | 0.9630 | 32 | sample_trust_20260504_025508_mlp_final_layer_align_bs32_lr0p001_seed0 |
| mlp | final_layer_align_mag | 4 | 0.9610 | 0.9635 | 32 | sample_trust_20260504_025508_mlp_final_layer_align_mag_bs32_lr0p001_seed0 |
| mlp | inverse_loss | 4 | 0.9567 | 0.9595 | 32 | sample_trust_20260504_025508_mlp_inverse_loss_bs32_lr0p001_seed0 |
| mlp | none | 4 | 0.9604 | 0.9645 | 16 | sample_trust_20260504_025508_mlp_none_bs16_lr0p001_seed0 |

## By Batch Size and Trust Mode

| batch_size | mode | runs | mean_best_test_acc | mean_final_test_acc | mean_final_trust_mean |
| --- | --- | --- | --- | --- | --- |
| 16 | combined_simple | 2 | 0.9718 | 0.9703 | 0.1218 |
| 16 | confidence | 2 | 0.9698 | 0.9647 | 0.9860 |
| 16 | ema_final_layer_align | 2 | 0.9718 | 0.9652 | 0.1152 |
| 16 | entropy | 2 | 0.9718 | 0.9683 | 0.9784 |
| 16 | final_layer_align | 2 | 0.9723 | 0.9715 | 0.1684 |
| 16 | final_layer_align_mag | 2 | 0.9710 | 0.9665 | 0.1512 |
| 16 | inverse_loss | 2 | 0.9703 | 0.9703 | 0.9862 |
| 16 | none | 2 | 0.9728 | 0.9715 | 1.0000 |
| 32 | combined_simple | 2 | 0.9675 | 0.9667 | 0.1074 |
| 32 | confidence | 2 | 0.9688 | 0.9680 | 0.9848 |
| 32 | ema_final_layer_align | 2 | 0.9703 | 0.9663 | 0.1256 |
| 32 | entropy | 2 | 0.9730 | 0.9690 | 0.9763 |
| 32 | final_layer_align | 2 | 0.9752 | 0.9598 | 0.1544 |
| 32 | final_layer_align_mag | 2 | 0.9723 | 0.9708 | 0.1257 |
| 32 | inverse_loss | 2 | 0.9698 | 0.9665 | 0.9845 |
| 32 | none | 2 | 0.9715 | 0.9560 | 1.0000 |
| 64 | combined_simple | 2 | 0.9677 | 0.9630 | 0.1016 |
| 64 | confidence | 2 | 0.9660 | 0.9640 | 0.9829 |
| 64 | ema_final_layer_align | 2 | 0.9627 | 0.9590 | 0.1480 |
| 64 | entropy | 2 | 0.9705 | 0.9705 | 0.9768 |
| 64 | final_layer_align | 2 | 0.9698 | 0.9650 | 0.1538 |
| 64 | final_layer_align_mag | 2 | 0.9715 | 0.9705 | 0.1167 |
| 64 | inverse_loss | 2 | 0.9700 | 0.9695 | 0.9823 |
| 64 | none | 2 | 0.9705 | 0.9688 | 1.0000 |
| 128 | combined_simple | 2 | 0.9688 | 0.9670 | 0.0974 |
| 128 | confidence | 2 | 0.9640 | 0.9635 | 0.9756 |
| 128 | ema_final_layer_align | 2 | 0.9587 | 0.9585 | 0.1540 |
| 128 | entropy | 2 | 0.9695 | 0.9695 | 0.9621 |
| 128 | final_layer_align | 2 | 0.9683 | 0.9667 | 0.1725 |
| 128 | final_layer_align_mag | 2 | 0.9708 | 0.9708 | 0.1166 |
| 128 | inverse_loss | 2 | 0.9640 | 0.9637 | 0.9758 |
| 128 | none | 2 | 0.9685 | 0.9685 | 1.0000 |

## Delta vs Matched `none` Baseline

| model | batch_size | seed | mode | best_test_acc | baseline_best_test_acc | delta |
| --- | --- | --- | --- | --- | --- | --- |
| cnn | 32 | 0 | final_layer_align | 0.9875 | 0.9830 | +0.0045 |
| mlp | 128 | 0 | final_layer_align_mag | 0.9595 | 0.9555 | +0.0040 |
| cnn | 16 | 0 | final_layer_align | 0.9845 | 0.9810 | +0.0035 |
| mlp | 32 | 0 | final_layer_align_mag | 0.9635 | 0.9600 | +0.0035 |
| mlp | 32 | 0 | final_layer_align | 0.9630 | 0.9600 | +0.0030 |
| mlp | 128 | 0 | entropy | 0.9585 | 0.9555 | +0.0030 |
| cnn | 32 | 0 | entropy | 0.9850 | 0.9830 | +0.0020 |
| cnn | 16 | 0 | entropy | 0.9830 | 0.9810 | +0.0020 |
| cnn | 16 | 0 | final_layer_align_mag | 0.9830 | 0.9810 | +0.0020 |
| cnn | 16 | 0 | combined_simple | 0.9830 | 0.9810 | +0.0020 |
| mlp | 128 | 0 | final_layer_align | 0.9575 | 0.9555 | +0.0020 |
| cnn | 16 | 0 | inverse_loss | 0.9825 | 0.9810 | +0.0015 |
| cnn | 64 | 0 | inverse_loss | 0.9810 | 0.9795 | +0.0015 |
| cnn | 64 | 0 | final_layer_align | 0.9810 | 0.9795 | +0.0015 |
| cnn | 64 | 0 | final_layer_align_mag | 0.9810 | 0.9795 | +0.0015 |
| mlp | 128 | 0 | combined_simple | 0.9570 | 0.9555 | +0.0015 |
| mlp | 32 | 0 | entropy | 0.9610 | 0.9600 | +0.0010 |
| cnn | 128 | 0 | final_layer_align_mag | 0.9820 | 0.9815 | +0.0005 |
| cnn | 16 | 0 | confidence | 0.9815 | 0.9810 | +0.0005 |
| cnn | 64 | 0 | combined_simple | 0.9800 | 0.9795 | +0.0005 |
| mlp | 64 | 0 | final_layer_align_mag | 0.9620 | 0.9615 | +0.0005 |
| cnn | 16 | 0 | ema_final_layer_align | 0.9810 | 0.9810 | +0.0000 |
| cnn | 64 | 0 | confidence | 0.9795 | 0.9795 | +0.0000 |
| cnn | 64 | 0 | entropy | 0.9795 | 0.9795 | +0.0000 |
| mlp | 64 | 0 | entropy | 0.9615 | 0.9615 | +0.0000 |
| cnn | 32 | 0 | ema_final_layer_align | 0.9825 | 0.9830 | -0.0005 |
| mlp | 32 | 0 | inverse_loss | 0.9595 | 0.9600 | -0.0005 |
| cnn | 128 | 0 | entropy | 0.9805 | 0.9815 | -0.0010 |
| cnn | 128 | 0 | combined_simple | 0.9805 | 0.9815 | -0.0010 |
| cnn | 64 | 0 | ema_final_layer_align | 0.9785 | 0.9795 | -0.0010 |

Interpretation note: this report is descriptive. Treat any improvement as a hypothesis until it survives repeated seeds and a full-data run.
