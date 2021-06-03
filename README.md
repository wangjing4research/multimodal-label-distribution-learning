# Introduction

This project proposes the multimodal label distribution learning (MLDL) framework for multimodal machine learning. Besides the multi-modalities, we consider the overall situation which will influence the weight of each modality in fusion. MLDL firstly recovers the multimodal label distribution (MLD) by leveraging side-information, then fuses the multi-modalities with the guidance of MLD and learns from the jointly representation. We design two models named early fusion MLDL (EF-MLDL) and late fusion (LF-MLDL) to deal with the sequential data.

# Get Start

1. Prepare Virtual Environment:
- Python >= 3.5
- Tensorflow >= 2.0.0
- sklearn >= 0.19
- gensim >= 3.8

2. Download Supported publicly availabel Datasets:
- [MOSI & MOSEI](https://github.com/A2Zadeh/CMU-MultimodalSDK)
- [MIMIC-III](https://mimic.physionet.org/)
- [CCS single-level diagnosis](1https://www.hcup-us.ahrq.gov/toolssoftware/ccs/
Multi_Level_CCS_2015.zip)

3. Process Datasets:
- MOSI & MOSEI:
	` python process_sentiment.py mosi 0 ./cmumosi`
	- `mosi` for MOSI dataset while `mosei` for MOSEI dataset
	- `0` for multimodal features, else for side-information
	- `./cmumosi` for the path to store data
- MIMIC-III:
	` python process_disease.py ./data_raw ./data_pro`
	- `./data_raw` for the path of folder that contains the raw data
	- `./data_pro` for the path of folder that stores the processed data
- Your own dataset:
	You can prepare your own dataset containing the side-information and the multimodal features.

4. Demo Scripts:
- EF-MLDL: `python EF_MLDL_demo.py`
- LF-MLDL: `python LF-MLDL_demo.py`
