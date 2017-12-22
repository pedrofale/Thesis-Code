# Code for my Msc Thesis
Focusing on the analysis of single-cell RNA-seq data, the methods used aim at disentangling the biological factors of variation from the technical. This disentanglement yields more reliable downstream analysis of the data, such as Differential Expression analysis between two groups of cells.

The starting paper for my research is "Dirichlet Process Mixture Model for Correcting Technical Variation in Single-Cell Gene Expression Data" by Prabhakaran et al, 2016. The idea there is to model the data using an infinite mixture model with extra parameters accounting for technical variation, which are estimated simultaneously with the other mixture parameters.

In `IIEEC/` you can find the code used for the experiments ran for the IIEEC report prior to the start of the thesis itself, where I applied Neural Network-based Autoencoders to extract features from bulk RNA-seq data useful for predicting breast cancer presence.
