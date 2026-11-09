# How to Run CaPT on Your PC (Easy Guide)

This guide explains, step by step, how to run the **CaPT** project.
It is written for a beginner on **Windows**.

> For what each file does, read [document.md](document.md).
> The official short instructions are in [CaPT/README.md](../CaPT/README.md).

---

## Table of Contents

1. [Read This First: What Your PC Can Do](#1-read-this-first-what-your-pc-can-do)
2. [Choose Your Path](#2-choose-your-path)
3. [Path A: Small Test on Your Own PC (CPU)](#3-path-a-small-test-on-your-own-pc-cpu)
4. [Path B: Real Training on Google Colab (Free GPU)](#4-path-b-real-training-on-google-colab-free-gpu)
5. [Path C: A Computer with an NVIDIA GPU](#5-path-c-a-computer-with-an-nvidia-gpu)
6. [How to Read the Results](#6-how-to-read-the-results)
7. [Problems and Fixes](#7-problems-and-fixes)
8. [Quick Checklist](#8-quick-checklist)

---

## 1. Read This First: What Your PC Can Do

This project **trains** a language model. Training needs a lot of computing power.
The authors used an **NVIDIA GPU with 48 GB of memory**.

When this guide was written, your PC had:

| Item | Your PC | What CaPT wants | OK? |
| --- | --- | --- | --- |
| Operating system | Windows 10 Pro | Linux (Windows works with some changes) | Partly |
| Graphics card | **Intel HD Graphics 630** | **NVIDIA GPU** with CUDA | **No** |
| Python | Not installed | Python 3.9 | Install it |
| conda | Not installed | Miniconda or Anaconda | Install it |

**What this means:**

- The code **can** run without an NVIDIA GPU. It then uses the **CPU** (the normal processor).
- On the CPU, training is **very slow**. A small model on a small dataset is fine for a test. The full experiments from the paper are not practical.
- An Intel graphics card **cannot** be used by this code. Only NVIDIA cards work (PyTorch uses NVIDIA's CUDA).

You can check your graphics card yourself: press `Win + R`, type `dxdiag`, press Enter, and open the **Display** tab.

[Back to top](#table-of-contents)

---

## 2. Choose Your Path

| Path | Where it runs | Cost | Speed | Good for |
| --- | --- | --- | --- | --- |
| **A. Your PC (CPU)** | Your computer | Free | Very slow | Learning how the code works, a quick test |
| **B. Google Colab** ⭐ | Google's computers, in your browser | Free (with time limits) | Fast | Real training on T5-Base |
| **C. NVIDIA GPU computer** | A gaming PC, lab PC, or rented cloud GPU | Free to paid | Fast | All experiments, including Llama and Qwen |

**Recommendation:** Do **Path A** first to learn the steps, then use **Path B** for real results.

[Back to top](#table-of-contents)

---

## 3. Path A: Small Test on Your Own PC (CPU)

You will do this once. It takes about 30–60 minutes, mostly waiting for downloads.

### Step 1: Install Miniconda

Miniconda gives you Python and a tool (`conda`) to keep this project's packages separate from everything else.

1. Go to <https://docs.anaconda.com/miniconda/> and download the **Windows 64-bit** installer.
2. Run it. Keep the default options ("Just Me" is fine).
3. When it finishes, open the Start menu and search for **"Anaconda Prompt (miniconda3)"**. Open it.

> Use **Anaconda Prompt** for all the commands below, not the normal PowerShell.
> It already knows where `conda` and `python` are.

Check that it works:

```bat
conda --version
```

You should see something like `conda 24.x.x`.

### Step 2: Go to the project folder

```bat
d:
cd "D:\AI Projects\capt\CaPT"
```

**Every command after this must be run from inside the `CaPT` folder.** The scripts look for files like `data/superglue/...` starting from where you are.

### Step 3: Create a Python 3.9 environment

```bat
conda create -n capt python=3.9.18 -y
conda activate capt
```

After `conda activate capt`, the start of the line shows `(capt)`. That means you are inside the project's environment.

> Each time you open a new Anaconda Prompt, run `conda activate capt` again.

### Step 4: Install PyTorch (CPU version)

The normal `requirements.txt` asks for NVIDIA and Linux-only packages (`nvidia-*`, `triton`). These **fail on Windows**. So we install the CPU version of PyTorch by hand:

```bat
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

### Step 5: Install the other packages

Make a copy of the package list **without** the packages that do not work on Windows.
Run this one line:

```bat
findstr /V /B /C:"nvidia-" /C:"triton" /C:"torch==" /C:"bitsandbytes" requirements.txt > requirements-windows.txt
```

Then install from the new list:

```bat
pip install -r requirements-windows.txt
pip install sentencepiece
```

(`bitsandbytes` is listed in the file but the training code does not use it, so it is safe to skip.)

### Step 6: Install the CaPT version of `transformers`

The project includes its **own modified copy** of the Hugging Face `transformers` library. The CaPT code lives inside it. Install it:

```bat
python setup.py install
```

Check that the correct version is installed:

```bat
python -c "from transformers.models.t5.modeling_t5 import CaPT; print('CaPT found')"
```

You must see `CaPT found`.

> ⚠️ **Never** run `pip install transformers` in this environment. It replaces the CaPT version with the normal one, and the training will break.

### Step 7: Download the datasets

```bat
python superglue/get_huggingface_superglue.py
```

This downloads 6 small datasets (BoolQ, CB, COPA, MultiRC, RTE, WiC) and saves them in `data\superglue\`.

> The README says `data/superglue/get_huggingface_superglue.py`. That path is wrong for this folder. The correct one is `superglue/get_huggingface_superglue.py`.

### Step 8: Run a small test

Use **COPA**, which is the smallest dataset, and small settings so the CPU can handle it.
Copy the whole command as **one line**:

```bat
python capt/train_t5.py --method gen-prompt --generator capt --model_name_or_path t5-base --tokenizer_name_or_path t5-base --dataset_name copa --num_virtual_tokens 1 --prompt_layers all --lr 1 --batch_size 4 --epoch 2 --max_length 128 --warmup_ratio 0.01 --max_patience 20 --random_seed 88
```

What changed from the paper's settings, and why:

| Option | Paper | Test on CPU | Why |
| --- | --- | --- | --- |
| `--batch_size` | 32 | 4 | Less memory and less work per step |
| `--epoch` | 50 | 2 | Finish in a reasonable time |
| `--max_length` | 512 | 128 | COPA sentences are short, so this is enough |

The first run also downloads the T5-Base model (about 0.9 GB).

**What you should see:**

- A line with the number of trainable parameters. For T5-Base with CaPT it should be about **9,216 (≈0.004%)**. This shows CaPT is working.
- A progress bar for each epoch.
- At the end: `Test result | Score : XX.XX`.

With only 2 epochs the score will be low. That is normal. The goal here is to see that everything runs.

**Even faster test:** replace `t5-base` with `t5-small` in both places (`--model_name_or_path` and `--tokenizer_name_or_path`). It is a smaller model, so it is quicker, but it is not a model from the paper.

**Do not try Llama or Qwen on your PC.** They are 1–1.5 billion parameters and are far too slow on a CPU.

[Back to top](#table-of-contents)

---

## 4. Path B: Real Training on Google Colab (Free GPU)

Google Colab is a website where you run Python on Google's computers, with a free NVIDIA GPU (usually a **T4 with 16 GB**). You only need a Google account and a browser.

### Step 1: Open Colab and turn on the GPU

1. Go to <https://colab.research.google.com> and click **New notebook**.
2. Menu: **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save**.

### Step 2: Upload the project

1. Make a ZIP file of the `CaPT` folder on your PC (right-click the folder → **Send to → Compressed (zipped) folder**).
2. In Colab, click the **folder icon** on the left, then the **upload** icon, and upload `CaPT.zip`.

Alternative: download it directly from GitHub inside Colab (step 3 shows both).

### Step 3: Run these cells

Paste each block into a new code cell and press `Shift + Enter`.

**Cell 1: Get the code and check the GPU**

```python
# If you uploaded CaPT.zip:
!unzip -q CaPT.zip -d /content/
# OR, if you prefer GitHub, use this line instead of the one above:
# !git clone https://github.com/comeandcode/CaPT.git /content/CaPT

%cd /content/CaPT
!nvidia-smi
```

`nvidia-smi` should show a Tesla T4 card.

**Cell 2: Install packages**

Colab already has PyTorch with GPU support, and it uses a newer Python than 3.9. So we do **not** use `requirements.txt`. We install only what is needed:

```python
!pip uninstall -y transformers
!pip install "numpy<2" datasets==2.20.0 tokenizers==0.19.1 huggingface-hub==0.24.2 accelerate==0.33.0 tensorboardX scikit-learn sentencepiece setuptools
!pip install .
```

`pip install .` installs the CaPT version of `transformers` from this folder.

If Colab shows a **"Restart session"** button after this cell, click it, then run `%cd /content/CaPT` again before continuing.

**Cell 3: Check CaPT and download the data**

```python
%cd /content/CaPT
!python -c "from transformers.models.t5.modeling_t5 import CaPT; print('CaPT found')"
!python superglue/get_huggingface_superglue.py
```

**Cell 4: Train T5-Base on COPA (paper settings)**

```python
!python capt/train_t5.py --method gen-prompt --generator capt \
  --model_name_or_path t5-base --tokenizer_name_or_path t5-base \
  --dataset_name copa --num_virtual_tokens 1 --prompt_layers all \
  --lr 1 --batch_size 32 --epoch 50 --max_length 512 \
  --warmup_ratio 0.01 --max_patience 20 --random_seed 88
```

To train on another dataset, change `--dataset_name` to `boolq`, `cb`, `multirc`, `rte_superglue` or `wic`.

**Cell 5: Save your results to your PC**

```python
!zip -r results.zip results log
```

Then download `results.zip` from the folder panel on the left.

### Things to know about Colab

- **Files are deleted** when the session ends. Download your results before closing.
- The free version **stops after a few hours** or if you are idle. Start with small datasets (COPA, CB, RTE). BoolQ and MultiRC are bigger and take longer.
- If you get `CUDA out of memory`, change `--batch_size 32` to `--batch_size 16` or `8`.
- Colab's Python is newer than the 3.9 the authors used. The cells above handle the known differences. If an install step fails, see [Problems and Fixes](#7-problems-and-fixes).

[Back to top](#table-of-contents)

---

## 5. Path C: A Computer with an NVIDIA GPU

Use this if you get access to a PC with an NVIDIA card, a university server, or a rented cloud GPU (for example RunPod, Vast.ai, Lambda, or a cloud provider). A Linux machine is easiest, because the original `requirements.txt` was made for Linux.

On a **Linux** machine with an NVIDIA GPU, follow the official steps:

```bash
cd CaPT
conda create -n capt python=3.9.18 -y
conda activate capt
pip install -r requirements.txt
python setup.py install
python superglue/get_huggingface_superglue.py
sh train.sh
```

`train.sh` runs three examples one after another: T5 on COPA, Llama on CB, and Qwen on CB.

**About the models:**

| Model | GPU memory needed (about) | Extra step |
| --- | --- | --- |
| T5-Base | 8–16 GB | None |
| T5-Large | 24 GB or more | None |
| Qwen-2.5-1.5B | 24 GB or more | None |
| Llama-3.2-1B | 16–24 GB or more | You must get access first (see below) |

These are rough numbers. If memory runs out, lower `--batch_size`.

**Llama access:**

1. Make an account on <https://huggingface.co>.
2. Open <https://huggingface.co/meta-llama/Llama-3.2-1B> and accept the license. Wait for approval.
3. Create an access token: **Settings → Access Tokens → New token** (type "Read").
4. On the training machine, run `huggingface-cli login` and paste the token.

**On a Windows PC with an NVIDIA GPU:** follow Path A, but in Step 4 install the GPU version of PyTorch instead:

```bat
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

Then check: `python -c "import torch; print(torch.cuda.is_available())"` must print `True`.

[Back to top](#table-of-contents)

---

## 6. How to Read the Results

After training you get three kinds of output, all inside the `CaPT` folder:

| Where | What is inside |
| --- | --- |
| The terminal | For each epoch: train loss, validation loss, validation score. At the end: `Test result \| Score : XX.XX` |
| `results\<model>\<dataset>\*.txt` | The final test score and how long training took |
| `log\` | Training curves for TensorBoard |
| `t5_models\` | The best saved model (T5 only) |

**The number that matters** is the final **test score**. The code reports the test score from the epoch that had the best validation score.

**To see the training curves:**

```bat
pip install tensorboard
tensorboard --logdir log
```

Open the link it prints (usually <http://localhost:6006>) in your browser.

**What scores to expect (T5-Base, full 50 epochs):** compare with Table 1 of the paper. Your numbers can be a few points different because of hardware, random seed, and package versions. With only 2 epochs on the CPU, scores will be much lower. That is expected.

[Back to top](#table-of-contents)

---

## 7. Problems and Fixes

| Problem | Why it happens | What to do |
| --- | --- | --- |
| `'conda' is not recognized` | You are in normal PowerShell or CMD | Open **Anaconda Prompt (miniconda3)** from the Start menu |
| `'python' is not recognized` or it opens the Microsoft Store | Python is not installed, or the environment is not active | Open Anaconda Prompt and run `conda activate capt` |
| `pip install -r requirements.txt` fails with `triton` or `nvidia-...` | These packages exist only for Linux | On Windows use `requirements-windows.txt` from [Path A, Step 5](#step-5-install-the-other-packages) |
| `ImportError: cannot import name 'CaPT'` or `... has no attribute 'prompt_len'` | The normal `transformers` is installed instead of the CaPT version | `pip uninstall -y transformers`, then `python setup.py install` (or `pip install .` on Colab) inside `CaPT` |
| `FileNotFoundError: data/superglue/...` | Data not downloaded, or you are in the wrong folder | `cd` into the `CaPT` folder, then run `python superglue/get_huggingface_superglue.py` |
| Data download fails and mentions `trust_remote_code` | Newer rules on Hugging Face | Open [get_huggingface_superglue.py](../CaPT/superglue/get_huggingface_superglue.py) and change `load_dataset('super_glue', dataset)` to `load_dataset('super_glue', dataset, trust_remote_code=True)` |
| `Wrong dataset name 'rte'` | The code calls this dataset `rte_superglue` | Use `--dataset_name rte_superglue` |
| `ModuleNotFoundError: No module named 'distutils'` | Python 3.12 or newer | On your PC use Python 3.9. On Colab run `!pip install setuptools` |
| `CUDA_VISIBLE_DEVICES=0 is not recognized` | That is Linux syntax | On Windows just leave it out and start with `python ...` |
| `sh train.sh` does not work on Windows | `.sh` is a Linux script | Copy one command from [train.sh](../CaPT/train.sh) and run it without the `CUDA_VISIBLE_DEVICES=0` part |
| Warnings like `torch.cuda.amp ... CUDA is not available` | You are on the CPU | Harmless. Training continues on the CPU |
| `CUDA out of memory` | The GPU is too small for these settings | Use a smaller `--batch_size` (16, 8, 4) or a smaller `--max_length` |
| Training is extremely slow | It is running on the CPU | Expected on your PC. Use Colab (Path B) for real training |
| Llama download gives 401 or 403 | Llama needs permission | See "Llama access" in [Path C](#5-path-c-a-computer-with-an-nvidia-gpu) |
| Qwen trains without prompts | `train_qwen.py` has a different default | Always add `--generator capt` |
| Command breaks into pieces in Anaconda Prompt | The `\` line breaks are for Linux and Colab | On Windows, put the whole command on **one line** |

[Back to top](#table-of-contents)

---

## 8. Quick Checklist

**On your PC (Path A):**

- [ ] Miniconda installed, and Anaconda Prompt opens
- [ ] `cd "D:\AI Projects\capt\CaPT"`
- [ ] `conda create -n capt python=3.9.18 -y` and `conda activate capt`
- [ ] CPU PyTorch installed
- [ ] `requirements-windows.txt` made and installed, plus `sentencepiece`
- [ ] `python setup.py install`, and the check prints `CaPT found`
- [ ] Datasets downloaded into `data\superglue\`
- [ ] Small COPA test runs and prints about 9,216 trainable parameters

**On Colab (Path B):**

- [ ] GPU turned on (T4)
- [ ] Project uploaded or cloned, `%cd /content/CaPT`
- [ ] Packages installed, and the check prints `CaPT found`
- [ ] Datasets downloaded
- [ ] Training started with paper settings
- [ ] `results.zip` downloaded before the session ends

[Back to top](#table-of-contents)
