## Project overview

- Aim: develop a Multi-view deconfounding VAE (multi-view data integration + confounder correction)
- Data: 
    - Rotterdam study
        - 500 individuals
        - Cardiovascular diseases
        - Methylation
        - 3D facial images
    - Toy data (TCGA)
        - 3024 patients
        - 6 cancers
        - 2000 most variable mRNAs
        - 1000 most variable miRNAs
- Conducted by Sonja Katz and Zuqi Li
- Supervised by Prof. Kristel Van Steen, Dr. Gennady Roshchupkin and Prof. Vitor Martins Dos Santos
- Google folder: https://drive.google.com/drive/folders/1GwZbMpVWW4xqdxmw_JRq9-DAR0WlnkE4

## Installation

```bash
## cd Multi-view-Deconfounding-VAE
conda env create -f environment.yml
source activate env_multiviewVAE
```

## Workplan

- [x] 1. Select basic model
    - Simidjievski, Nikola, et al. "Variational autoencoders for cancer data integration: design principles and computational practice." Frontiers in genetics 10 (2019): 1205.
    - https://github.com/CancerAI-CL/IntegrativeVAEs
    - The X-shaped Variational Autoencoder (X-VAE) Architecture is overall recommended in this comparative study <br/><img src="https://user-images.githubusercontent.com/7692477/233080494-22abb000-8def-4ddb-b9a2-fa2a582392d2.png" width="500">
- [x] 2. Reform the basic model
    - [x] Implement in Pytorch Lightning
    - [x] Rearrange code
    - [x] Provide two latent loss functions (KL divergence and Maximum Mean Discrepancy)
    - [x] Implement testing metrics
- [x] 3. Create a clustering model
    - [x] Strategy 1: Run K-means (or other clustering methods) on the latent space
    - [ ] Strategy 2: Add a term in the loss function to iteratively optimize the clustering quality
- [ ] 4. Correct for confounders
    - [ ] Strategy 1: Take confounders into account during decoding and the loss function is conditioned on the confounders; Adapted from: Lawry Aguila, Ana, et al. "Conditional VAEs for Confound Removal and Normative Modelling of Neurodegenerative Diseases." Medical Image Computing and Computer Assisted Intervention–MICCAI 2022: 25th International Conference, Singapore, September 18–22, 2022, Proceedings, Part I. Cham: Springer Nature Switzerland, 2022. <br/><img src="https://user-images.githubusercontent.com/7692477/226375457-f5d7bd2b-7b79-4b8f-83c3-e3a696ad200f.png" width="500">
      - [ ] Build cVAE - concat covariates to input dim & latent dim
    
    - [ ] Strategy 2: Add a term in loss function to minimize the association/similarity between the latent embedding and confounders <br/><img src="https://user-images.githubusercontent.com/7692477/233090210-96ab3edd-3cc5-4c79-b291-8c761d6214ee.png" width="500">
      - [ ] Talk again with Lau about his idea + implementation (as easy as correlating latent features with covariates and adding it to the loss?)






