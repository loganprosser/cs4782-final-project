# Kalman Trust Optimizer Summary

## Ranked Variants
- `old_kalman_grad_scaling`: accuracy `0.9793` +/- `0.0021` over `3` runs; NLL `0.0694`, ECE `0.0203`.
- `adamw_clip`: accuracy `0.9770` +/- `0.0018` over `3` runs; NLL `0.0734`, ECE `0.0207`.
- `adamw`: accuracy `0.9770` +/- `0.0015` over `3` runs; NLL `0.0778`, ECE `0.0216`.
- `per_unit_filter_trust`: accuracy `0.9762` +/- `0.0004` over `3` runs; NLL `0.0749`, ECE `0.0233`.
- `kalman_lr_controller`: accuracy `0.9549` +/- `0.0020` over `3` runs; NLL `0.1503`, ECE `0.0404`.
- `combined_kalman_trust`: accuracy `0.9542` +/- `0.0029` over `3` runs; NLL `0.1510`, ECE `0.0406`.
- `direction_aware_trust`: accuracy `0.9345` +/- `0.0025` over `3` runs; NLL `0.2249`, ECE `0.0535`.
