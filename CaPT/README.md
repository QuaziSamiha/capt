# All You Need is One: Capsule Prompt Tuning with a Single Vector

🔥🔥🔥 Our [arxiv](https://arxiv.org/abs/2510.16670) version is currently available. Please check it out! 🔥🔥🔥

<div align="center">
  <img src=".\images\capt_overview.png">
</div>
<p align="center">
<b>Figure1: Overview of Deep Prompt-Tuning vs. CaPT (ours) Frameworks. (a) Original Deep
Prompt-Tuning. (b) The overall architecture of our proposed CaPT, integrating both
task-aware guidance and instance-aware signal to trigger “attention anchor”.</b>
</p>

## Installation

```bash
git clone https://github.com/comeandcode/CaPT.git
cd CaPT
conda create -n capt python=3.9.18 -y
conda activate capt
pip install -r requirements.txt
python setup.py install
```
## Download Datasets
```bash
python data/superglue/get_huggingface_superglue.py
```


## Training & Evaluation

```bash
sh train.sh
```

## Citation
```bash
@inproceedings{liu2025all,
  title={All You Need is One: Capsule Prompt Tuning with a Single Vector},
  author={Liu, Yiyang and Liang, James Chenhao and Fan, Heng and Yang, Wenhao and Cui, Yiming and Han, Xiaotian and Huang, Lifu and Liu, Dongfang and Wang, Qifan and Han, Cheng},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
}
```


## Acknowledgments
Our training pipeline and data processing module are implemented based on code base [SMoP](https://github.com/jyjohnchoi/SMoP). We thank the authors for their effort.