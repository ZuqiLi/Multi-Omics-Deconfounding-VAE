## Project overview

- Aim: develop a Multi-omics deconfounding VAE (multi-omics data integration + confounder correction)
- Data: 
    - Rotterdam study
        - 500 individuals
        - Cardiovascular diseases
        - Methylation
        - 3D facial images
    - Toy data (TCGA)
        - 2547 patients
        - 6 cancers
        - 2000 most variable mRNAs
        - 2000 most variable DNAm
- Conducted by Sonja Katz and Zuqi Li
- Supervised by Prof. Kristel Van Steen, Dr. Gennady Roshchupkin and Prof. Vitor Martins Dos Santos
- Google folder: https://drive.google.com/drive/folders/1GwZbMpVWW4xqdxmw_JRq9-DAR0WlnkE4

## Installation

```bash
## cd Multi-view-Deconfounding-VAE
conda env create -f environment.yml
source activate env_multiviewVAE
```



## Overview models:

- all models are in folder: `models`
- optimal architecture from `modelOptimisation` experiments: `latentSize = 50`; `hiddenlayers = 200`

- **XVAE**: 
    - `XVAE` - Simidjievski, Nikola, et al. "Variational autoencoders for cancer data integration: design principles and computational practice." Frontiers in genetics 10 (2019): 1205.

<br/><img src="https://user-images.githubusercontent.com/7692477/233080494-22abb000-8def-4ddb-b9a2-fa2a582392d2.png" width="300">

- **cXVAE**:
    1. input+embed: 
    2. input: 
    3. embed: 
    4. fused+embed: 

- **Adversarial Training**
    1. XVAE with one adversarial network and multiclass predicition: `adversarial_XVAE_multiclass`
        - `XVAE_adversarial_multiclass`: inspired by [Dincer et al.](https://academic.oup.com/bioinformatics/article/36/Supplement_2/i573/6055930); training over all batches
        - `XVAE_adversarial_1batch_multiclass`: original by [Dincer et al.](https://academic.oup.com/bioinformatics/article/36/Supplement_2/i573/6055930)

        <br/><img src="https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/36/Supplement_2/10.1093_bioinformatics_btaa796/6/m_btaa796f3.jpeg?Expires=1690948061&Signature=ungPUfIvRqzHnmz8feJh4Fh4Eu8j5HR-5~-7Q8u6AnA0S72qXlsZKBGergKaJjzY-yXMy0KDK4ufIqoqHFH3Di~WHzh0w9Rf1jK-n-hzHDZ3VtUtUiZFcPYKgkOwXOE5NJzvj9sqoG02oa3yhqFGxNFlufHrshinJzeKaEMMw0XqAT2RjXWQ41kzZZRsp8iM2dnD-jdloyN6EvJ4CONDrSd5llCQaw0S9GgLCCqy5iPTEDi6QTRc3YILo2KUX8XjvYcjx4xuLfFRvTy2lhouVHpPyUDF0-v6oyhXMFuTZl54b2OPI02fXsyosY91MDRB1XdTzDdVrW-NW15QKgpOFw__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA" width="300">

        - `XVAE_scGAN_multiclass`: inspired by [Bahrami et al.](https://academic.oup.com/bioinformatics/article/37/10/1345/5998665)

        <br/><img src="https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/37/10/10.1093_bioinformatics_btaa976/2/m_btaa976f1.jpeg?Expires=1691396622&Signature=n4sxy7kREFNARc8qbBOyvJMYHZDvGFquXsTWbzkCwsLXM5WjtJNCiz72JupH6vqaJypFr4jvdIE3ZlIFd20GExBxy7pSxzMQNuTFu1MMb3Wy0IKxzmN9EfMRNmxQQT9hrAb0gdmaxKfWqaBsb5ktSc0Ms4VzUl2Xa8YkOi2MHN5--qc7rHryTcO1~rZU3~Nf3qr4Sw2jHHxj7YIlV740fYypQMIua~ifGr0gc94L355iqoTjcYBzY9nHI6o1fXGkhtcJOv7uc2nmfIAKON7N~nxvJX4lbKStbfGQq~UPUn3Shbz3IELaw61x1bgAnZar8Zm-Bg5fzhHzfEKLbzo31w__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA" width="300">


    2. XVAE with multiple adversarial network (one for each confounder): `adversarial_XVAE_multipleAdvNet`

          - XAE with multiple adversarial network (outdated?)






## Workplan

- [x] 1. Select basic model
    - Simidjievski, Nikola, et al. "Variational autoencoders for cancer data integration: design principles and computational practice." Frontiers in genetics 10 (2019): 1205.
    - https://github.com/CancerAI-CL/IntegrativeVAEs
    - The X-shaped Variational Autoencoder (X-VAE) Architecture is overall recommended in this comparative study 
- [x] 2. Reform the basic model
    - [x] Implement in Pytorch Lightning
    - [x] Rearrange code
    - [x] Provide two latent loss functions (KL divergence and Maximum Mean Discrepancy)
    - [x] Implement testing metrics
- [x] 3. Create a clustering model
    - [x] Strategy 1: Run K-means (or other clustering methods) on the latent space
    - [ ] Strategy 2: Add a term in the loss function to iteratively optimize the clustering quality
- [ ] 4. Correct for confounders
    - [x] Strategy 1: Take confounders into account during decoding and the loss function is conditioned on the confounders; Adapted from: Lawry Aguila, Ana, et al. "Conditional VAEs for Confound Removal and Normative Modelling of Neurodegenerative Diseases." Medical Image Computing and Computer Assisted Intervention–MICCAI 2022: 25th International Conference, Singapore, September 18–22, 2022, Proceedings, Part I. Cham: Springer Nature Switzerland, 2022. <br/><img src="https://user-images.githubusercontent.com/7692477/226375457-f5d7bd2b-7b79-4b8f-83c3-e3a696ad200f.png" width="500">
      - [x] Build cVAE - concat covariates to input dim & latent dim
    
    - [ ] Strategy 2: Add a term in loss function to minimize the association/similarity between the latent embedding and confounders <br/><img src="https://user-images.githubusercontent.com/7692477/233090210-96ab3edd-3cc5-4c79-b291-8c761d6214ee.png" width="500">
      - [ ] XVAE with adversarial training; [Inspiration from this paper](https://academic.oup.com/bioinformatics/article/36/Supplement_2/i573/6055930)
        - [x] simple version; only pre-training
        - [ ] ping pong training

    - [ ] Strategy 3: Simply remove confounded latent features






