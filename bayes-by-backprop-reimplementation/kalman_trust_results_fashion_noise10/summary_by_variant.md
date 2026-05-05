# Kalman Trust Optimizer Summary

## Ranked Variants
- `per_unit_filter_trust`: accuracy `0.8747` +/- `0.0116` over `3` runs; NLL `0.4183`, ECE `0.1006`.
- `adamw`: accuracy `0.8733` +/- `0.0098` over `3` runs; NLL `0.4276`, ECE `0.1013`.
- `old_kalman_grad_scaling`: accuracy `0.8731` +/- `0.0073` over `3` runs; NLL `0.4264`, ECE `0.0957`.
- `adamw_clip`: accuracy `0.8717` +/- `0.0096` over `3` runs; NLL `0.4271`, ECE `0.0993`.
- `combined_kalman_trust`: accuracy `0.8607` +/- `0.0134` over `3` runs; NLL `0.4682`, ECE `0.1127`.
- `kalman_lr_controller`: accuracy `0.8545` +/- `0.0105` over `3` runs; NLL `0.4935`, ECE `0.1211`.
- `direction_aware_trust`: accuracy `0.8442` +/- `0.0103` over `3` runs; NLL `0.5360`, ECE `0.1344`.
