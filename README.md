# AgentRX: Multimodal Agentic Benchmark

Table of Contents
=================

<!--ts-->
  * [Background](#Background)
  * [Overview](#Overview)
  * [Environment Setup](#Environment-Setup)
  * [Dataset](#Dataset)
  * [Model Training](#Model-Training)
  * [Model Evaluation](#Model-Evaluation)
  * [Citation](#Citation)
<!--te-->


Background
============
we introduce **AgentRx**, a comprehensive evaluation framework to analyze agent performance across three progressive settings. 
First, we establish performance baselines using only a single modality, specifically clinical notes encompassed within patient summaries. 
Second, we assess the capacity of a single agent to synthesize heterogeneous multimodal data within one context window, utilizing modality dropping ablations to measure robustness. 
Finally, we investigate whether multi-agent reasoning across specialized agents can improve performance compared to single agent approaches.


Setup
====================================
We build on the MIMIC-IV, MIMIC-CXR, and MIMIC-Note datasets for our experiments. 

Environment Setup
==================
```bash
git clone https://github.com/your-org/AgentRx.git
cd AgentRx
python3 -m venv venv_name
source venv_name/bin/activate
pip install -r requirements.txt
```

Dataset
-------------
We use the following datasets for our experiments:

- [MIMIC-IV EHR](https://physionet.org/content/mimiciv/1.0/)
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- [MIMIC-IV Notes](https://physionet.org/content/mimic-iv-note/2.2/)

Please follow the [README](mimic4extract/README.md) in `mimic4extract/` for extracting and preparing the time-series EHR data. 

Before running scripts, make sure to create the dataset json file.

To resize chest X-ray images:
```bash
python resize.py
```

To ensure consistent splits between CXR and EHR datasets:
```bash
python create_split.py
```

Refer to `arguments.py` for full configuration options.

Then run the jsonl creation script

```bash
python updated_dataset_creator.py     --mimic_iv_dir data/mimic-iv-extracted/root/     --mimic_cxr_dir data/physionet.org/files/mimic-cxr-jpg/2.0.0/     --mimic_notes_dir /MIMIC-Note/physionet.org/files/mimic-iv-note/2.2/note/     --ehr_data_dir data/mimic-iv-extracted/     --cxr_image_dir data/physionet.org/files/mimic-cxr-jpg/2.0.0/resized/    --output_dir ./multimodal_dataset_splits/
```

Model Inference
-----------------
Feel free to run any of the scripts inside the scripts folder.

```bash
# Train Each unimodal encoder for mortality
sh ./scripts/mortality/Unimodal/Qwen/CoT.sh
```


