# Figure Captions

## MNIST Test Error Curves
This plot reproduces the paper's MNIST learning-curve comparison by tracking test error across training for deterministic SGD, dropout, Bayes by Backprop, and our Bayes by Backprop + Dropout extension. It was made to show whether Bayes by Backprop follows dropout-like generalization behavior during training. Lower curves indicate better test performance.

## Final Test Error Bar Plot
This bar plot compares the best test error reached by each of the four reproduced methods. It was made to summarize the final accuracy story in one poster-friendly view. The conclusion is that Bayes by Backprop is competitive with dropout when their bars are close.

## Paper Vs Ours Comparison Table
This table places our MNIST test errors next to the closest Table 1 values from Blundell et al. It was made to separate qualitative reproduction from exact numeric matching. Differences should be interpreted in light of our smaller training budget and implementation details.

## Weight Histograms
The histogram compares deterministic trained weights with Bayesian posterior means from the saved checkpoints, mirroring the paper's Figure 3-style weight-density diagnostic.

## Training Runtime Comparison
This plot compares total wall-clock training time for the four methods. It was made to quantify the computational cost of Bayesian weight sampling and posterior parameters. Higher bars indicate more expensive training.

## Peak Memory Comparison
This plot compares memory use for each method. It was made to show whether the Bayesian methods require extra memory for posterior means and variances. Higher bars indicate a larger memory footprint. The original CPU peak metric was not method-isolated because `psutil` was unavailable and all methods ran in one Python process. For the poster-facing memory comparison, we therefore plot checkpoint/model parameter memory, which captures the expected approximately 2x parameter footprint of Bayes by Backprop.

## Runtime Vs Accuracy Tradeoff
This scatter plot shows best test error against total runtime. It was made to ask whether a method's accuracy is worth its training cost. Methods closest to the lower-left are the most efficient.

## Memory Vs Accuracy Tradeoff
This scatter plot shows best test error against the memory metric used for the comparison. It was made to ask whether a method's accuracy is worth its memory footprint. Methods closest to the lower-left are the most memory-efficient. The original CPU peak metric was not method-isolated because `psutil` was unavailable and all methods ran in one Python process. For the poster-facing memory comparison, we therefore plot checkpoint/model parameter memory, which captures the expected approximately 2x parameter footprint of Bayes by Backprop.

## Reproduction Dashboard
This combined figure collects the core reproduction evidence: learning curves, final errors, runtime, memory, and paper-vs-ours values. It was made as a single poster-ready summary. The main conclusion is visible by comparing Bayes by Backprop with dropout and then checking whether the extension improves the tradeoff.
