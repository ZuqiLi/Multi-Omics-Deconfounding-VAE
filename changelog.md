# Recent changes / updates

[SK] - 27.03.2023: 
- added `environment.yml`
- re-arranged code into smaller modules
- added test step to be used for final quality checks after training (loss, R2, relative error, clustering metric)


[SK] - 28.04.2023:
- added reconstruction accuracy metrices for scoring (table in tensorboard)
- added embedding view in tensorboard

[Zl] - 15.05.2023:
- onehot encode stage and race, rescale age to [0,1]
- shuffle and split confounders together with train/val/test sets
- test both original version and deconfounding version of XVAE on the data
- add associations between final clustering and covariates as metrics
- try to add cov in the input layer instead of fused layer


[SK] - 22.05.2023:
- implemented adversarial training (no ping pong); `trainModel_adversarial.py`

[SK] - 04.06.2023:
- implemented adversarial training for multiple covariates (ping pong); `adversarial_XVAE_multiCov.py`