# Final Noisy Multi-Seed Comparison With External Kalman

| Model | Runs | Mean Error | Std Error | Mean Accuracy | Std Accuracy | Best Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Standard MLP h400/L2 | 3 | 0.1192 | 0.0028 | 0.8808 | 0.0028 | 0.8828 |
| Standard MLP + Kalman h400/L2 | 3 | 0.1223 | 0.0037 | 0.8777 | 0.0037 | 0.8799 |
| Standard MLP h800/L2 | 3 | 0.1199 | 0.0034 | 0.8801 | 0.0034 | 0.8824 |
| Standard MLP + Kalman h800/L2 | 3 | 0.1192 | 0.0017 | 0.8808 | 0.0017 | 0.8821 |
| Dropout MLP h400/L2 | 3 | 0.1323 | 0.0014 | 0.8677 | 0.0014 | 0.8686 |
| Dropout MLP + Kalman h400/L2 | 3 | 0.1346 | 0.0015 | 0.8654 | 0.0015 | 0.8671 |
| BBB h400/L2 | 3 | 0.1231 | 0.0027 | 0.8769 | 0.0027 | 0.8787 |
| BBB + Kalman h400/L2 | 3 | 0.1237 | 0.0008 | 0.8763 | 0.0008 | 0.8772 |
| BBB + Dropout h400/L2 | 3 | 0.1239 | 0.0008 | 0.8761 | 0.0008 | 0.8769 |
| BBB + Dropout + Kalman h400/L2 | 3 | 0.1248 | 0.0002 | 0.8752 | 0.0002 | 0.8753 |
| Standard MLP + kalmanalgo tensor h400/L2 | 1 | 0.1225 | 0.0000 | 0.8775 | 0.0000 | 0.8775 |
| Standard MLP + kalmanalgo tensor h800/L2 | 1 | 0.1246 | 0.0000 | 0.8754 | 0.0000 | 0.8754 |
