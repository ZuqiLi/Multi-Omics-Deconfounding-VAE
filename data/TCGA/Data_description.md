### Step 1. Download data from TCGA
- R package `TCGAbiolinks`
- 3024 patients (after step 2) with 6 cancer types:
    - BRCA (951)
    - THCA (408)
    - BLCA (387)
    - LUSC (364)
    - HNSC (412)
    - KIRC (502)
- mRNA expression profiles (normalized by reads per million)
- miRNA expression profiles (normalized by reads per million)
- Clinical data: 
    - tumor stage: i, ia, ib, ii, iia, iib, iii, iiia, iiib, iiic, iv, iva, ivb, ivc, x
    - age at diagnosis
    - race: 'white', 'black or african amarican', 'asian', 'american indian or alaska native'
    - gender

### Step 2. Removal criteria
- Patients with NA or 'not reported' clinical data
- race 'american indian or alaska native'
- tumor stage x
- mRNA or miRNA with 0 variance

### Step 3. Encode clinical vairables and save datasets
- mRNA dataset: 3024 patients x 58,456 mRNAs
- miRNA dataset: 3024 patients x 1695 miRNAs
- clinic dataset: 3024 patients x 6 variables
    1. patient ID
    2. tumor stage: 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4
    3. age at diagnosis
    4. race: asian(1), black or african amarican(2), white(3)
    5. gender: female(0), male(1)
    6. cancer type: BRCA(1), THCA(2), BLCA(3), LUSC(4), HNSC(5), KIRC(6)
    
### Step 4. Pre-process the datasets
- mRNA dataset: 'TCGA_mRNAs_processed.csv'
    - L2 normalization for every mRNA
    - Take the 2000 mRNAs with highest variance
    - 3024 patients x 2000 mRNAs
- miRNA dataset: 'TCGA_miRNAs_processed.csv'
    - L2 normalization for every miRNA
    - Take the 1000 miRNAs with highest variance
    - 3024 patients x 1000 miRNAs
- clinic dataset: 'TCGA_clinic.csv'
    
