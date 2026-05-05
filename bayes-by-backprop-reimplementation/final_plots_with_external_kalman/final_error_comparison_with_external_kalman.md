# Final Multi-Seed Comparison With External Kalman

| Model | Runs | Mean Error | Std Error | Mean Accuracy | Std Accuracy | Best Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Standard MLP h400/L2 | 3 | 0.0207 | 0.0018 | 0.9793 | 0.0018 | 0.9811 |
| Standard MLP + Kalman h400/L2 | 3 | 0.0206 | 0.0010 | 0.9794 | 0.0010 | 0.9803 |
| Dropout MLP h400/L2 | 3 | 0.0195 | 0.0006 | 0.9805 | 0.0006 | 0.9810 |
| Dropout MLP + Kalman h400/L2 | 3 | 0.0202 | 0.0011 | 0.9798 | 0.0011 | 0.9810 |
| BBB h400/L2 | 3 | 0.0198 | 0.0004 | 0.9802 | 0.0004 | 0.9807 |
| BBB + Kalman h400/L2 | 3 | 0.0195 | 0.0010 | 0.9805 | 0.0010 | 0.9816 |
| BBB + Dropout h400/L2 | 3 | 0.0182 | 0.0004 | 0.9818 | 0.0004 | 0.9821 |
| BBB + Dropout + Kalman h400/L2 | 3 | 0.0182 | 0.0003 | 0.9818 | 0.0003 | 0.9821 |
| Standard MLP h800/L2 | 5 | 0.0201 | 0.0018 | 0.9799 | 0.0018 | 0.9824 |
| Standard MLP + Kalman h800/L2 | 5 | 0.0200 | 0.0019 | 0.9800 | 0.0019 | 0.9829 |
| Standard MLP + kalmanalgo tensor h800/L2 | 1 | 0.0174 | 0.0000 | 0.9826 | 0.0000 | 0.9826 |
| BBB + Dropout + kalmanalgo tensor h400/L2 | 1 | 0.0193 | 0.0000 | 0.9807 | 0.0000 | 0.9807 |
