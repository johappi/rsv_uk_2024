# Parental intent to accept infant RSV monoclonal antibody immunisation in the UK: local trends, social and behavioural determinants, and strategies to maximise population protection


This repository contains code for fitting a series of Bayesian models related to RSV (respiratory syncytial virus) immunization intent in the UK population. The models use both original survey data and safeguarded census microdata, along with MRP (multilevel regression and post-stratification) to estimate intent across socio-demographic groups.

---

## Models

- **Model A**  
  Hypothetical RSV intent *pre* is modeled as an ordinal logistic regression in sociodemographic variables, where the pre indicates that the corresponding question is asked before the randomized control trial (RCT). Sociodemographics are aligned (coarsened) to the combined microsamples for post-stratification, which is the main objective of Model A

- **Model B**  
  Same as Model A, except that the sociodemographic predictors are not aligned to the microsamples, which allows finer categories, particularly for Ethnicity.

- **Model C**  
  Builds on Model B by adding *Behavioral and Social Drivers* (BeSD) as predictors. Model C helps in understanding the impact of BeSD on immunisation intent and, by comparison to Model B, how BeSD are possibly mediating the effects of sociodemographics on immunisation intent

- **Model D**  
  Builds on Model A and includes the *Randomized Control Trial* (RCT), that is, Hypothetical RSV intent post is added as additional dependent variable. Sociodemographics are aligned to microsamples for post-stratification. Model D examines how information about the timing and importance of the immunisation causally impacts immunisation intent.

- **Model E**  
  Builds on a slightly simplified version of Model D and includes RSV intent for own child as additional dependent variable. While post-stratification is not feasible for this model, the sociodemographics are aligned to microsamples to keep the number of categories lower, as the sample size (number of respondents with a child aged 0 or 1) is relatively small. Model E examines whether the immunisation intent for parent’s own actual child differs from the previously asked immunisation intent for a hypothetical child.

---

## Repository Structure

### `scripts/`
Python scripts for data alignment, inference, and post-stratification:
- `region_mappings.py` — maps postcodes to ITL regions.
- `microsamples_eng_wales.py`, `microsamples_scot.py`, `microsamples_ni.py` — align microsamples to the survey.
- `questionnaire.py` — aligns both survey and microsamples to ITL regions for UK, England and Wales, Scotland, and NI.
- `inference.py` — main script for running inference; imports model specifications from `src/fit/`.
- `poststratify.py` — post-stratification for UK models.
- `poststratify_causal_varying_c.py` — post-stratification for RCT-based UK models.


### `src/`

Python source code for model fitting and plotting.

- `src/fit/`:  
  Defines `fit_stan()` functions for Models A–E. These functions:
  - prepare model-specific data based on survey and microsamples,  
  - load the corresponding Stan model from `stan_codes/`,  
  - run HMC sampling (via CmdStanPy),  
  - and save results as arviz idata.

  Includes:
  - `uk.py` – Models A, D, E  
  - `ew.py` – Models B, C (fitted to whole UK, sociodemographics aligned to microsamples in England and Wales)  
  - `ni.py`, `scot.py` – for completeness, not used in this study

- `src/plot/`:  
  Plotting functions

### `notebooks/`
Jupyter notebooks for producing plots and tables used in the paper.

### `stan_codes/`
Stan code for all models:
- `mrp_ord_uk.stan` – Model A
- `ord_ew_wo_micro.stan` – Model B
- `bes_soc_ord_ew.stan` – Model C
- `ord_causal_soc_hier_uk.stan` – Model D
- `ord_causal_child_soc_cond_on_pre_and_post.stan` – Model E

Models not used in this study:
- `mrp_ber_ni.stan`, `mrp_ber_scot.stan`, `mrp_ber_uk.stan`, `ord_causal_wo_socio.stan`, `mrp_ord_uk_w_enp.stan`, `mrp_ord_ew.stan`, `ord_causal_child_soc_cond_on_pre.stan`, `ord_causal_child_soc_cond_on_post.stan`

### `dat/questionnaire/`
Contains the original survey file:
- `RSV-September14-final.csv`

---

## Data Access

Due to licensing restrictions, safeguarded microdata samples cannot be shared in this repository. However, they are publicly available upon request from the UK Data Service:

- **England & Wales (2021)**  
  [DOI: 10.5255/UKDA-SN-9155-1](http://doi.org/10.5255/UKDA-SN-9155-1)

- **Scotland (2011)**  
  [DOI: 10.5255/UKDA-SN-7835-1](http://doi.org/10.5255/UKDA-SN-7835-1)

- **Northern Ireland (2011)**  
  [DOI: 10.5255/UKDA-SN-7770-1](http://doi.org/10.5255/UKDA-SN-7770-1)

The geospatial shapefiles, lookup tables, and other census tables used in the analysis are publicly available and can be shared upon request.

---

## Repository and Reproducibility Notes

- Some folders like `idata/` (inference outputs), `post_strat/` (post-stratified posterior samples)  `dat/geo_boundaries`, `dat/region_mappings`, `dat/census_2021`, `dat/safeguarded_microdata_2011`, `dat/safeguarded_microdata_2021` (raw data),  are excluded via `.gitignore` to avoid uploading large or private files.
- All scripts assume a project root directory is set via the environment variable `WORK_DIR`. If not set, a default relative path is used.
- Most scripts depend on the safeguarded microdata and will not run without access to those files. While the structure is modular, certain processing steps assume the presence of these data.

---

## License

This repository is made available for academic transparency and reproducibility. See the `LICENSE` file for more information.

---

## Citation

If you use this code, please cite the accompanying paper (citation details will be added here upon publication).

---

## Repository

📦 Code: [https://github.com/johappi/rsv_uk_2024](https://github.com/johappi/rsv_uk_2024)

