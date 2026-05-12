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
| Standard MLP + kalmanalgo tensor h800/L2 | 3 | 0.0183 | 0.0015 | 0.9817 | 0.0015 | 0.9828 |
| Standard MLP + kalmanalgo scalar h800/L2 | 3 | 0.0195 | 0.0015 | 0.9805 | 0.0015 | 0.9820 |
| BBB + Dropout + kalmanalgo scalar h400/L2 | 3 | 0.0199 | 0.0014 | 0.9801 | 0.0014 | 0.9814 |
| BBB + Dropout + kalmanalgo tensor h400/L2 | 3 | 0.0216 | 0.0010 | 0.9784 | 0.0010 | 0.9794 |
