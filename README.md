# Multi-omics Deconfounding Variational Autoencoder

![fig3-02](https://github.com/user-attachments/assets/9eba9db0-bd98-4fca-8889-ca44b93121e5)

MODVAE (**M**ulti-**O**mics **D**econfounding **V**ariational **A**uto**E**coder) is a project proposing four novel VAE-based deconfounding frameworks tailored for clustering multi-omics data. These frameworks effectively mitigate confounding effects while preserving genuine biological patterns. The deconfounding strategies employed include (A) a conditional VAE, (B) adversarial training, (C) adding a regularization term to the loss function, and (D) removal of latent features correlated with confounders. For all models, we use a X-shaped architecture to merge the heterogeneous input data sources into a combined latent representation [1]. Consensus clustering is applied to the latent representation generated by each VAE-based model.
Details of this project can be found in our publication [2].

## Environment
For better reproducibility, it's recommended to refer to the following hardware and software settings:
- Operating system: Ubuntu 20.04.6 LTS
- Processor: Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz
- Memory: 767 GiB
- Graphics: llvmpipe (LLVM 12.0.0, 256 bits)
- Python version: 3.9.7

The required packages can be installed with the conda environment file in this repository:
```
## cd Multi-view-Deconfounding-VAE
conda env create -f environment.yml
source activate env_multiviewVAE
```

## Overview models:

- all models are in folder: `models`
- optimal architecture from `modelOptimisation` experiments: `latentSize = 50`; `hiddenlayers = 200`

- **XVAE**: 
    - `XVAE` - Simidjievski, Nikola, et al. [1]

<br/><img src="https://user-images.githubusercontent.com/7692477/233080494-22abb000-8def-4ddb-b9a2-fa2a582392d2.png" width="300">

- **cXVAE**:
    1. `cXVAE.cXVAE_inputEmbed`: condition confounders on both the input and embedding layers
    2. `cXVAE.cXVAE_input`: condition confounders on the input layer
    3. `cXVAE.cXVAE_embed`: condition confounders on the embedding layer
    4. `cXVAE.cXVAE_fusedEmbed`: condition confounders on both the fusion and the embedding layers

- **Adversarial Training**
    1. XVAE with one adversarial network and multiclass predicition: `adversarial_XVAE_multiclass`
        - `XVAE_adversarial_multiclass`: inspired by Dincer et al. [3]; training over all batches
        - `XVAE_adversarial_1batch_multiclass`: original by Dincer et al. [3] \
![m_btaa796f3](https://github.com/user-attachments/assets/a287b681-757b-4289-bc11-cbe6ce79522c)
        - `XVAE_scGAN_multiclass`: inspired by Bahrami et al. [4] \
![m_btaa976f1](https://github.com/user-attachments/assets/be639762-6dcf-41be-8767-0bc217735527)

    2. XVAE with multiple adversarial network (one for each confounder): `adversarial_XVAE_multipleAdvNet`
 
- **Regularization to loss function**
    1. `XVAE_corrReg`: inspired by Liu et al. [5]
    2. Possibile regularization functions
       1. 'corrAbs': absolute Pearson correlation
       2. 'corrSq': squared Pearson correlation
       3. 'MIhist': mutual information implemented with histogram
       4. 'MIkde': mutual information implemented with KDE
 
- **Removal of latent features**
    1. that are correlated with confounders based on Pearson correlation
    2. that are statistically associated with confounders based on P-value

## Tutorial
To use our MODVAE models, please download/clone the whole repository from Github to your local. \
For the best-performing model, cXVAE, we've created a wrapper file `cXVAE_wrapper.py` for easy usage. It can be run via the following command on terminal:
```
python  cXVAE_wrapper.py  pathname  filename_view1  filename_view2  filename_label  conf_type  filename_conf
```
### Parameters:
- `pathname`: The first dataset, a 2D numpy array of size $p1 \times p2$.
- `filename_view1`: The second dataset, a 3D numpy array of size $p1 \times p3 \times p4$.
- `filename_view2`: The rank of $G_1$, a positive integer value.
- `filename_label`: The rank of $G_2$, a positive integer value.
- `conf_type`: The confounder type, which is a string out of 'conti' for continuous variable, 'categ' for categorical variable, and 'multi' for multiple variables.
- `filename_conf`: The rank of $G_4$, a positive integer value.
### Output:
- `G1`: A nonnegative numpy array of size $p1 \times r1$ with `float32` data type.
- `G2`: A nonnegative numpy array of size $p2 \times r2$ with `float32` data type.
- `G3`: A nonnegative numpy array of size $p3 \times r3$ with `float32` data type.
- `G4`: A nonnegative numpy array of size $p4 \times r4$ with `float32` data type.

&nbsp;
```
find_best_r1(R12, R13, r1_list, r2, r3, r4, n_init=10, stop=200)
```
### Description:
Find the best value for `r1`, the rank of $G_1$. This function runs INMTD with random initialization for multiple (`n_init`) times with different `r1` values. In each repitition, a clustering of samples is derived by assigning each sample to the cluster with highest value in the corresponding column of $G_1$. Note that the number of columns of $G_1$, namely `r1`, is the same as number of clusters. Subsequently, a consensus clustering is calculated from the ensemble of clusterings with the same `r1`, yielding a stability score. The `r1` value with highest stability score is chosen as the best.
### Parameters:
- `R12`: The first dataset, a 2D numpy array of size $p1 \times p2$.
- `R13`: The second dataset, a 3D numpy array of size $p1 \times p3 \times p4$.
- `r1_list`: A list of positive integer values for the rank of $G_1$ to be tested.
- `r2`: A positive integer value for the rank of $G_2$.
- `r3`: A positive integer value for the rank of $G_3$.
- `r4`: A positive integer value for the rank of $G_4$.
- `n_init`: A positive integer value for the number of repititions of random initialization. The default is 10.
- `stop`: A positive integer value for the maximal number of iterations that INMTD runs in each repitition. The default is 200.
### Returns:
- A positive integer value for the best `r1`.

&nbsp;
```
INMTD(R12, R13, r1, r2, r3, r4, init='svd', stop=500)
```
### Description:
Run the INMTD model to joint decompose 2D and 3D datasets.
### Parameters:
- `R12`: The first dataset, a 2D numpy array of size $p1 \times p2$.
- `R13`: The second dataset, a 3D numpy array of size $p1 \times p3 \times p4$.
- `r1`: A positive integer value for the rank of $G_1$.
- `r2`: A positive integer value for the rank of $G_2$.
- `r3`: A positive integer value for the rank of $G_3$.
- `r4`: A positive integer value for the rank of $G_4$.
- `init`: A string for the initialization method. Possible values are 'random', 'svd', and 'rsvd'. The default is 'svd'.
- `stop`: A positive integer value for the maximal number of iterations that INMTD runs. The default is 500.
### Returns:
- `embedding`: A list containing embedding matrices $G_1$, $G_2$, $G_3$, $G_4$, the core matrix $S_{12}$ for `R12`, and the core tensor $\mathcal{S}_{13}$ for `R13`.
- `logging`: A 2D numpy array with 6 columns corresponding to the joint reconstruction error of `R12` and `R13`, the reconstruction error of `R12`, the reconstruction error of `R13`, the joint relative error of `R12` and `R13`, the relative error of `R12`, and the relative error of `R13`. Rows are the recording of the 6 metrics in the first 10 iterations and every 10 iterations afterwards.
  
## Example
Here is an example of how to run INMTD with simulated data. Functions and example datasets of the simulation can be found in the `Simulation` folder.
### Simulation
```
from simulation import generate_data


p1, p2, p3, p4 = 1000, 250, 80, 20
r1, r2, r3, r4 = 5, 10, 4, 2

R12, R13, clust1, clust2, clust3, clust4 = generate_data([p1, p2, p3, p4], [r1, r2, r3, r4])
```
The `generate_data` function in `simulation.py` takes 2 parameters as input. The first parameter is a list containing the numbers of dimensions of the 2 datasets to be simulated, namely `R12` and `R13`. The second parameter is another list containing the ranks of the embedding matrices composing `R12` and `R13`. It returns the 2D matrix `R12`, the 3D tensor `R13`, the true clustering on the dimension of `p1`, the true clustering on the dimension of `p2`, the true clustering on the dimension of `p3`, and the true clustering on the dimension of `p4`.

### INMTD pipeline
```
from INMTD import INMTD
import numpy as np


embedding, logging = INMTD(R12, R13, r1, r2, r3, r4)
print(logging)
```

## References
> [1] Simidjievski N, Bodnar C, Tariq I, Scherer P, Andres Terre H, Shams Z, Jamnik M and Liò P (2019) Variational Autoencoders for Cancer Data Integration: Design Principles and Computational Practice. *Frontiers in Genetics*, 10:1205. doi: 10.3389/fgene.2019.01205 \
> [2] Zuqi Li, Sonja Katz, Edoardo Saccenti, David W Fardo, Peter Claes, Vitor A P Martins dos Santos, Kristel Van Steen, Gennady V Roshchupkin, Novel multi-omics deconfounding variational autoencoders can obtain meaningful disease subtyping, *Briefings in Bioinformatics*, Volume 25, Issue 6, November 2024, bbae512, doi: 10.1093/bib/bbae512 \
> [3] Ayse B Dincer, Joseph D Janizek, Su-In Lee, Adversarial deconfounding autoencoder for learning robust gene expression embeddings, Bioinformatics, Volume 36, Issue Supplement_2, December 2020, Pages i573–i582, doi: 10.1093/bioinformatics/btaa796 \
> [4] Mojtaba Bahrami, Malosree Maitra, Corina Nagy, Gustavo Turecki, Hamid R Rabiee, Yue Li, Deep feature extraction of single-cell transcriptomes by generative adversarial network, Bioinformatics, Volume 37, Issue 10, May 2021, Pages 1345–1351, doi: 10.1093/bioinformatics/btaa976 \
> [5] Xianjing Liu, Bo Li, Esther E. Bron, Wiro J. Niessen, Eppo B. Wolvius, and Gennady V. Roshchupkin. "Projection-wise disentangling for fair and interpretable representation learning: Application to 3d facial shape analysis." In Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part V 24, pp. 814-823. Springer International Publishing, 2021. doi: 10.1007/978-3-030-87240-3_78

 
