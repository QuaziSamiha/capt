# Running CaPT on Windows (CPU Test)

This guide shows how to run a **small test** of CaPT on your own Windows PC, using only the CPU.
No NVIDIA graphics card is needed.

The goal is to check that the code installs and runs. It is **not** meant to reproduce the paper's scores.
For real training, use Google Colab or an NVIDIA GPU (see [running.md](running.md), Paths B and C).

> For what each file in the project does, read [document.md](document.md).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Before You Start](#2-before-you-start)
3. [Step 1: Install Miniconda](#step-1-install-miniconda)
4. [Step 2: Open the Project Folder](#step-2-open-the-project-folder)
5. [Step 3: Create a Python 3.9 Environment](#step-3-create-a-python-39-environment)
6. [Step 4: Install PyTorch (CPU Version)](#step-4-install-pytorch-cpu-version)
7. [Step 5: Install the Other Packages](#step-5-install-the-other-packages)
8. [Step 6: Install the CaPT Version of transformers](#step-6-install-the-capt-version-of-transformers)
9. [Step 7: Download the Datasets](#step-7-download-the-datasets)
10. [Step 8: Run a Small Test on COPA](#step-8-run-a-small-test-on-copa)
11. [Step 9: Check the Results](#step-9-check-the-results)
12. [Problems and Fixes](#problems-and-fixes)
13. [All Commands in One Place](#all-commands-in-one-place)

---

## 1. Overview

You will do these steps once:

| Step | What you do | Why |
| --- | --- | --- |
| 1 | Install Miniconda | Gives you Python and `conda` |
| 2 | Go to the `CaPT` folder | All scripts expect to run from there |
| 3 | Create a Python 3.9 environment | The authors used Python 3.9.18 |
| 4 | Install the CPU version of PyTorch | Your PC has no NVIDIA GPU |
| 5 | Install the other packages | Using a Windows-safe copy of `requirements.txt` |
| 6 | Install the project's `transformers` | The CaPT code lives inside it |
| 7 | Download the data | Six SuperGLUE datasets |
| 8 | Run a short test on COPA | The smallest dataset, with smaller settings |
| 9 | Check the output | About **9,216 trainable parameters** means CaPT works |

Time needed: about 30–60 minutes, mostly waiting for downloads.

---

## 2. Before You Start

**What you need:**

- Windows 10 or 11, 64-bit
- About 10 GB of free disk space (Python packages, the T5-Base model, and the datasets)
- An internet connection
- At least 8 GB of RAM (16 GB is better)

**What to expect:**

- Training on a CPU is **slow**. The test in Step 8 uses small settings so it finishes in a reasonable time.
- An Intel or AMD graphics card **cannot** be used by this code. PyTorch only uses NVIDIA cards (through CUDA). Without one, it uses the CPU automatically.

**Where to type the commands:** use **Anaconda Prompt**, which you install in Step 1. Commands in this guide are for that window, not PowerShell.

[Back to top](#table-of-contents)

---

## Step 1: Install Miniconda

Miniconda is a small installer for Python. It also gives you `conda`, a tool that keeps each project's packages in a separate **environment**, so they do not clash with other software.

1. Go to <https://docs.anaconda.com/miniconda/> and download the **Windows 64-bit** installer.
2. Run it and keep the default options. "Just Me" is fine.
3. Open the Start menu, search for **Anaconda Prompt (miniconda3)**, and open it.

Check that `conda` works:

```bat
conda --version
```

**Expected output:** something like `conda 24.x.x`.

[Back to top](#table-of-contents)

---

## Step 2: Open the Project Folder

```bat
d:
cd "D:\AI Projects\capt\CaPT"
```

**Explanation:**

- `d:` switches to drive D. Anaconda Prompt starts on drive C, and `cd` alone does not change drives.
- The quotes are needed because the path has a space in it (`AI Projects`).

> **Important:** run every command after this from inside the `CaPT` folder.
> The scripts use paths like `data/superglue/copa`, which are read starting from the current folder.

[Back to top](#table-of-contents)

---

## Step 3: Create a Python 3.9 Environment

```bat
conda create -n capt python=3.9.18 -y
conda activate capt
```

**Explanation:**

- `conda create -n capt python=3.9.18` makes a new environment called `capt` with Python 3.9.18, the version the authors used. `-y` answers "yes" to the questions.
- `conda activate capt` turns the environment on. The start of the line now shows `(capt)`.

Check the Python version:

```bat
python --version
```

**Expected output:** `Python 3.9.18`

> Each time you open a new Anaconda Prompt, run `conda activate capt` again (and go to the `CaPT` folder again).

[Back to top](#table-of-contents)

---

## Step 4: Install PyTorch (CPU Version)

```bat
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

**Explanation:**

- PyTorch is the library that does the actual training.
- `torch==2.4.0` is the same version as in the project's `requirements.txt`.
- `--index-url .../whl/cpu` downloads the **CPU-only** build. It is much smaller than the GPU build and does not need NVIDIA drivers.

Check it:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Expected output:** `2.4.0+cpu False`. `False` is correct here: it means there is no GPU, so PyTorch will use the CPU.

[Back to top](#table-of-contents)

---

## Step 5: Install the Other Packages

The project's [requirements.txt](../CaPT/requirements.txt) was made on Linux with an NVIDIA GPU. Some lines in it **fail on Windows**:

| Lines in requirements.txt | Why they are removed |
| --- | --- |
| `nvidia-...` (12 packages) | NVIDIA CUDA libraries. Linux-only on pip, and not needed on CPU |
| `triton==3.0.0` | A GPU compiler. Linux-only |
| `torch==2.4.0` | Already installed in Step 4 (CPU build). Keeping it could pull the wrong build |
| `bitsandbytes==0.43.2` | A GPU library. The training code does not use it |

This one-line command makes a copy of the list **without** those lines. The original file is not changed.

```bat
findstr /V /B /C:"nvidia-" /C:"triton" /C:"torch==" /C:"bitsandbytes" requirements.txt > requirements-windows.txt
```

**Explanation of the command:**

| Part | Meaning |
| --- | --- |
| `findstr` | Windows tool that searches text in a file |
| `/V` | Keep the lines that do **not** match |
| `/B` | Match only at the **beginning** of a line |
| `/C:"..."` | One text to look for. There are four, one per package group |
| `requirements.txt` | The file to read |
| `> requirements-windows.txt` | Save the result into a new file |

Using `/B` matters: `torch==` must match only the line that **starts** with it, so packages like `tensorboardX` are kept.

Now install from the new list, plus `sentencepiece` (the T5 tokenizer needs it and it is not in the list):

```bat
pip install -r requirements-windows.txt
pip install sentencepiece
```

You can check the new file with `type requirements-windows.txt`. It should have no `nvidia-`, `triton`, `torch==` or `bitsandbytes` lines.

[Back to top](#table-of-contents)

---

## Step 6: Install the CaPT Version of transformers

The `CaPT` folder contains its **own modified copy** of the Hugging Face `transformers` library (version 4.43.2). The CaPT code is added inside it, for example the `CaPT` class in [modeling_t5.py](../CaPT/transformers/models/t5/modeling_t5.py). That is why `transformers` is commented out in `requirements.txt`.

Install it from the project folder:

```bat
python setup.py install
```

**Explanation:** [setup.py](../CaPT/setup.py) packages the local `transformers` folder and installs it into the `capt` environment, as if it were the normal library.

Check that the CaPT version is the one installed:

```bat
python -c "from transformers.models.t5.modeling_t5 import CaPT; print('CaPT found')"
```

**Expected output:** `CaPT found`

> ⚠️ **Never** run `pip install transformers` in this environment. It replaces the CaPT version with the normal one, and training breaks. If that happens, see [Problems and Fixes](#problems-and-fixes).

[Back to top](#table-of-contents)

---

## Step 7: Download the Datasets

```bat
python superglue/get_huggingface_superglue.py
```

**Explanation:** the script downloads six SuperGLUE datasets from Hugging Face (BoolQ, CB, COPA, MultiRC, RTE, WiC) and saves each split as a `.pkl` file:

```text
CaPT\data\superglue\
    boolq\    train.pkl  validation.pkl  test.pkl
    cb\       ...
    copa\     ...
    multirc\  ...
    rte\      ...
    wic\      ...
```

> The README says `python data/superglue/get_huggingface_superglue.py`. That path is wrong.
> The script is in `superglue\`, not `data\superglue\`. Use the command above.

Check that COPA was saved:

```bat
dir data\superglue\copa
```

You should see `train.pkl`, `validation.pkl` and `test.pkl`.

[Back to top](#table-of-contents)

---

## Step 8: Run a Small Test on COPA

COPA is the smallest dataset (400 training examples), so it is the best choice for a CPU test.

Copy this as **one line** (Anaconda Prompt does not support the `\` line breaks used on Linux):

```bat
python capt/train_t5.py --method gen-prompt --generator capt --model_name_or_path t5-base --tokenizer_name_or_path t5-base --dataset_name copa --num_virtual_tokens 1 --prompt_layers all --lr 1 --batch_size 4 --epoch 2 --max_length 128 --warmup_ratio 0.01 --max_patience 20 --random_seed 88
```

The first run also downloads the T5-Base model (about 0.9 GB) into the Hugging Face cache.

### What each option means

| Option | Value | Meaning |
| --- | --- | --- |
| `--method` | `gen-prompt` | Use the prompt-generation method (the only choice) |
| `--generator` | `capt` | Use CaPT to make the prompt |
| `--model_name_or_path` | `t5-base` | The base model. It stays frozen; only the prompt is trained |
| `--tokenizer_name_or_path` | `t5-base` | The tokenizer that matches the model |
| `--dataset_name` | `copa` | The dataset to train and test on |
| `--num_virtual_tokens` | `1` | One prompt vector, the "single vector" from the paper |
| `--prompt_layers` | `all` | Add the prompt at every layer |
| `--lr` | `1` | Learning rate. High, but normal for prompt tuning |
| `--batch_size` | `4` | Examples per training step |
| `--epoch` | `2` | Passes over the training data |
| `--max_length` | `128` | Maximum input length in tokens |
| `--warmup_ratio` | `0.01` | Raise the learning rate slowly for the first 1% of steps |
| `--max_patience` | `20` | Stop early after this many epochs with no improvement |
| `--random_seed` | `88` | Makes the run repeatable |

### What is smaller than in the paper, and why

The paper's settings are in [train.sh](../CaPT/train.sh). This test changes three of them:

| Option | Paper | This test | Why |
| --- | --- | --- | --- |
| `--batch_size` | 32 | 4 | Less memory and less work per step on the CPU |
| `--epoch` | 50 | 2 | Finishes in a reasonable time |
| `--max_length` | 512 | 128 | COPA inputs are short, so 128 tokens is enough and much faster |

The `CUDA_VISIBLE_DEVICES=0` part in `train.sh` is left out. It is Linux syntax and selects a GPU, which you do not have.

### Even faster: T5-Small

Replace `t5-base` with `t5-small` in **both** places. The model is smaller, so it runs faster, but it is not a model from the paper and the parameter count will be different.

> **Do not try Llama or Qwen on a CPU.** They have 1–1.5 billion parameters and are far too slow.

[Back to top](#table-of-contents)

---

## Step 9: Check the Results

### 1. Trainable parameters (the main check)

Near the start, the script prints:

```text
Trainable parameters: 9,216
Percentage of trainable parameters: 0.004...%
```

**If you see about 9,216 trainable parameters, CaPT is working.** This is the key result of the paper: only a tiny number of values are trained, and the rest of T5-Base (about 220 million parameters) stays frozen.

If the number is very different (for example millions), the normal `transformers` library is probably installed instead of the CaPT version. Go back to [Step 6](#step-6-install-the-capt-version-of-transformers).

### 2. Training progress

For each epoch you see a progress bar, then the train loss, validation loss and validation score.

Warnings such as `torch.cuda.amp ... CUDA is not available` are **harmless**. They only say that there is no GPU.

### 3. Final score

At the end:

```text
Test result | Score : XX.XX
```

With only 2 epochs the score will be low, often near 50 (COPA has two choices, so 50 is a coin flip). **That is expected.** This test checks that everything runs, not that it reaches the paper's score.

### 4. Output files

These appear inside the `CaPT` folder:

| Where | What is inside |
| --- | --- |
| `results\...\*.txt` | The final test score and the training time |
| `log\` | Training curves for TensorBoard |
| `t5_models\` | The best saved prompt/model |

To view the training curves:

```bat
pip install tensorboard
tensorboard --logdir log
```

Then open <http://localhost:6006> in your browser.

[Back to top](#table-of-contents)

---

## Problems and Fixes

| Problem | Why it happens | What to do |
| --- | --- | --- |
| `'conda' is not recognized` | You are in PowerShell or CMD | Open **Anaconda Prompt (miniconda3)** from the Start menu |
| `'python' is not recognized`, or the Microsoft Store opens | The environment is not active | Run `conda activate capt` |
| The prompt does not show `(capt)` | New window | Run `conda activate capt` and `cd` to the `CaPT` folder again |
| `pip install -r requirements.txt` fails on `triton` or `nvidia-...` | You used the original file | Use `requirements-windows.txt` from [Step 5](#step-5-install-the-other-packages) |
| `ImportError: cannot import name 'CaPT'` | The normal `transformers` is installed | `pip uninstall -y transformers`, then `python setup.py install` inside `CaPT` |
| `FileNotFoundError: data/superglue/...` | Data not downloaded, or wrong folder | `cd "D:\AI Projects\capt\CaPT"`, then repeat [Step 7](#step-7-download-the-datasets) |
| Download fails and mentions `trust_remote_code` | Newer Hugging Face rules | In [get_huggingface_superglue.py](../CaPT/superglue/get_huggingface_superglue.py), change `load_dataset('super_glue', dataset)` to `load_dataset('super_glue', dataset, trust_remote_code=True)` |
| `CondaToSNonInteractiveError: Terms of Service have not been accepted` | Newer conda asks you to accept Anaconda's terms first | Accept them with the `conda tos accept ...` commands it prints, or use the free conda-forge channel: `conda create -n capt python=3.9.18 -y --override-channels -c conda-forge` |
| `AttributeError: Can't pickle local object 'prepare_dataset.<locals>.collate_fn'` | Windows cannot start DataLoader worker processes for this code | In [train_t5.py](../CaPT/capt/train_t5.py), the three `DataLoader` lines use `num_workers=(0 if os.name == 'nt' else 4)` (already fixed in this copy) |
| `Invalid data_name 'rte'` | The code calls it `rte_superglue` | Use `--dataset_name rte_superglue` |
| `CUDA_VISIBLE_DEVICES=0 is not recognized` | Linux syntax | Leave it out; start the command with `python` |
| `sh train.sh` does not work | `.sh` is a Linux script | Copy the command from [Step 8](#step-8-run-a-small-test-on-copa) instead |
| The command breaks into pieces | `\` line breaks are for Linux | Put the whole command on one line |
| `MemoryError` or the PC freezes | Not enough RAM | Close other programs, use `--batch_size 2`, or try `t5-small` |
| Training is extremely slow | It runs on the CPU | Expected. Use Colab for real training ([running.md](running.md), Path B) |

[Back to top](#table-of-contents)

---

## All Commands in One Place

Run these in **Anaconda Prompt**, after installing Miniconda:

```bat
d:
cd "D:\AI Projects\capt\CaPT"

conda create -n capt python=3.9.18 -y
conda activate capt

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

findstr /V /B /C:"nvidia-" /C:"triton" /C:"torch==" /C:"bitsandbytes" requirements.txt > requirements-windows.txt
pip install -r requirements-windows.txt
pip install sentencepiece

python setup.py install
python -c "from transformers.models.t5.modeling_t5 import CaPT; print('CaPT found')"

python superglue/get_huggingface_superglue.py

python capt/train_t5.py --method gen-prompt --generator capt --model_name_or_path t5-base --tokenizer_name_or_path t5-base --dataset_name copa --num_virtual_tokens 1 --prompt_layers all --lr 1 --batch_size 4 --epoch 2 --max_length 128 --warmup_ratio 0.01 --max_patience 20 --random_seed 88
```

**Next time** you want to run it, you only need:

```bat
d:
cd "D:\AI Projects\capt\CaPT"
conda activate capt
python capt/train_t5.py ...
```

[Back to top](#table-of-contents)
