# Kalman Trust Optimizer Summary

## Ranked Variants
- `old_kalman_grad_scaling`: accuracy `0.8651` +/- `0.0000` over `1` runs; NLL `0.4367`, ECE `0.0946`.
- `adamw`: accuracy `0.8623` +/- `0.0000` over `1` runs; NLL `0.4522`, ECE `0.0988`.
- `per_unit_filter_trust`: accuracy `0.8613` +/- `0.0000` over `1` runs; NLL `0.4305`, ECE `0.0942`.
- `adamw_clip`: accuracy `0.8607` +/- `0.0000` over `1` runs; NLL `0.4361`, ECE `0.0952`.
- `combined_kalman_trust`: accuracy `0.8452` +/- `0.0000` over `1` runs; NLL `0.4989`, ECE `0.1125`.
- `kalman_lr_controller`: accuracy `0.8424` +/- `0.0000` over `1` runs; NLL `0.5255`, ECE `0.1235`.
- `direction_aware_trust`: accuracy `0.8324` +/- `0.0000` over `1` runs; NLL `0.5642`, ECE `0.1356`.
