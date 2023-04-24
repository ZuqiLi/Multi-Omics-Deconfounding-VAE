# Multi-view-Deconfounding-VAE
- Project by Zuqi Li and Sonja Katz
- Supervised by Prof. Kristel Van Steen, Dr. Gennady Roshchupkin and Prof. Vitor Martins Dos Santos

### Installation

```bash
## cd Multi-view-Deconfounding-VAE
conda env create -f environment.yml
source activate env_multiviewVAE
```

### Motivations
- Integration of multi-view data: Deep Learning (e.g. Autoencoders)
- Problem during model training: confounder adjustment (linear correction not sufficient!)
- Current research landscape: multitude of approaches, types of data used - do not necessarily overlap
- Challenges: number of input modalities, removal of all confounders, removal of unknown confounders, …

### Data
- Rotterdam study / Generation R
- Pittsburg study (Zuqi's data)
- Publicly available dataset (e.g. TCGA)

### VAE strategies
- Hira, Muta Tah, et al. "Integrated multi-omics analysis of ovarian cancer using variational autoencoders." Scientific reports 11.1 (2021): 6265.
![image](https://user-images.githubusercontent.com/7692477/226374761-c6fdf0b5-5b72-4bb6-9a0f-a0de72abb250.png)
- Ternes, Luke, et al. "A multi-encoder variational autoencoder controls multiple transformational features in single-cell image analysis." Communications biology 5.1 (2022): 255.
![image](https://user-images.githubusercontent.com/7692477/226374966-bbf783b9-8620-43a1-a0ad-7822bf52d42d.png)
- Yu, Tianwei. "AIME: Autoencoder-based integrative multi-omics data embedding that allows for confounder adjustments." PLoS Computational Biology 18.1 (2022): e1009826.
![image](https://user-images.githubusercontent.com/7692477/226375038-4cb86525-993a-4f53-ab33-f4d98b01b633.png)

### Deconfounding strategies
- Lawry Aguila, Ana, et al. "Conditional VAEs for Confound Removal and Normative Modelling of Neurodegenerative Diseases." Medical Image Computing and Computer Assisted Intervention–MICCAI 2022: 25th International Conference, Singapore, September 18–22, 2022, Proceedings, Part I. Cham: Springer Nature Switzerland, 2022.\
Yu, Tianwei. "AIME: Autoencoder-based integrative multi-omics data embedding that allows for confounder adjustments." PLoS Computational Biology 18.1 (2022): e1009826.
![image](https://user-images.githubusercontent.com/7692477/226375457-f5d7bd2b-7b79-4b8f-83c3-e3a696ad200f.png)
- Lau’s project on deconfounding autoencoder
![image](https://user-images.githubusercontent.com/7692477/226375544-dbda95dc-8f73-496c-9911-b41290491349.png)




