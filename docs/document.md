# CaPT Project Guide (Beginner Friendly)

A guide to the **CaPT** code folder for someone who is new to machine learning projects.
It explains what each file does, how the files work together, how the code matches the paper, and how to run it.

> **Paper:** *All You Need is One: Capsule Prompt Tuning with a Single Vector* (NeurIPS 2025)
> **Code folder:** [CaPT/](../CaPT/)
> **Easy paper summary:** [capt-summary/](../capt-summary/All%20You%20Need%20is%20One%20Capsule%20Prompt%20Tuning%20with%20a%20Single%20vector.md)

---

## Table of Contents

1. [The Big Idea in 2 Minutes](#1-the-big-idea-in-2-minutes)
2. [Words You Need to Know First](#2-words-you-need-to-know-first)
3. [Folder Map: What Is Where](#3-folder-map-what-is-where)
4. [What Each File Does](#4-what-each-file-does)
   - [Top-level files](#41-top-level-files)
   - [The capt folder: training scripts](#42-the-capt-folder-training-scripts)
   - [The src folder: data loaders](#43-the-src-folder-data-loaders)
   - [The config folder](#44-the-config-folder)
   - [The superglue folder: dataset downloader](#45-the-superglue-folder-dataset-downloader)
   - [The transformers folder: the modified library](#46-the-transformers-folder-the-modified-library)
   - [Folders created when you run the code](#47-folders-created-when-you-run-the-code)
5. [Architecture: How CaPT Is Built](#5-architecture-how-capt-is-built)
   - [The CaPT module (the core code)](#51-the-capt-module-the-core-code)
   - [How CaPT is plugged into T5](#52-how-capt-is-plugged-into-t5)
   - [How CaPT is plugged into Llama and Qwen](#53-how-capt-is-plugged-into-llama-and-qwen)
   - [Paper equation to code mapping](#54-paper-equation-to-code-mapping)
6. [Working Flow: What Happens When You Run It](#6-working-flow-what-happens-when-you-run-it)
   - [Step 1: Download data](#61-step-1-download-data)
   - [Step 2: Load and format data](#62-step-2-load-and-format-data)
   - [Step 3: Build the model](#63-step-3-build-the-model)
   - [Step 4: Freeze everything except CaPT](#64-step-4-freeze-everything-except-capt)
   - [Step 5: Training loop](#65-step-5-training-loop)
   - [Step 6: Evaluate, early stop and save](#66-step-6-evaluate-early-stop-and-save)
7. [Technology Used](#7-technology-used)
8. [Running Requirements](#8-running-requirements)
9. [How to Run: Step by Step](#9-how-to-run-step-by-step)
10. [Command-Line Options Explained](#10-command-line-options-explained)
11. [Common Problems and Fixes](#11-common-problems-and-fixes)
12. [Suggested Reading Order for the Code](#12-suggested-reading-order-for-the-code)

---

## 1. The Big Idea in 2 Minutes

**The problem:** Big language models (LLMs) are expensive to retrain. *Prompt-tuning* is a cheap shortcut: you **freeze** the whole model and train only a few extra vectors called **soft prompts**, which are placed in front of the input.

The paper says normal prompt-tuning has two weaknesses:
1. The soft prompts mostly look at each other and ignore the actual input sentence.
2. You must try many prompt lengths (5, 10, 20, 100 tokens…) to find the best one, which is slow.

**The CaPT solution:** Use **just ONE vector per layer** (the "capsule"). This vector is made of two parts added together:

```
capsule = (learned task vector)  +  (average of the current input's vectors)
           "what job to do"          "what this specific sentence is about"
           task-aware                instance-aware
```

Because the capsule contains information about the input, the input words pay attention to it and it pays attention back. The paper calls this an **"attention anchor"**.

**Result:** Similar or better accuracy than full fine-tuning on T5-Large, while training **at most 0.004%** of the model's parameters, and no prompt-length search.

> **In the code**, this whole idea is about 25 lines: the `CaPT` class in [modeling_t5.py:664-696](../CaPT/transformers/models/t5/modeling_t5.py#L664-L696). Everything else in the project is setup, data handling and the training loop.

[Back to top](#table-of-contents)

---

## 2. Words You Need to Know First

| Word | Simple meaning |
| --- | --- |
| **Model** | A big mathematical function with millions of numbers (parameters) that turns text into predictions. |
| **Parameter / weight** | One number inside the model that can be learned. |
| **Freeze** | Lock parameters so training does not change them. In PyTorch: `param.requires_grad = False`. |
| **Token** | A piece of text (a word or part of a word) turned into a number ID. |
| **Embedding / hidden state** | A list of numbers (a vector) that represents a token. T5-Base uses 768 numbers per token. Llama-3.2-1B uses 2048. |
| **Layer** | One processing step of a Transformer. T5-Base's encoder has 12 layers. Llama-3.2-1B has 16. |
| **Soft prompt** | An extra, trainable vector placed in front of the input tokens. |
| **Epoch** | One full pass over the training data. |
| **Batch** | A group of examples processed together, for example 32 sentences at once. |
| **Learning rate (lr)** | How big each training update step is. |
| **Loss** | A number that measures how wrong the model is. Training tries to make it smaller. |
| **Optimizer** | The algorithm that updates parameters to reduce the loss. This project uses **Adafactor**. |
| **Early stopping** | Stop training when the validation score has not improved for a while (the "patience"). |
| **Train / dev / test split** | Train = data the model learns from. Dev (validation) = data used to pick the best epoch. Test = final score. |
| **GPU / CUDA** | A graphics card and NVIDIA's software for running math fast. Training needs one. |
| **Hugging Face** | A website and Python library that provides pre-trained models and datasets. |
| **Encoder-decoder (T5)** | The model reads the input with an *encoder*, then writes the answer as text with a *decoder*. |
| **Decoder-only (Llama, Qwen)** | Models like ChatGPT that read text left-to-right. In this project they get a small classification layer on top. |

[Back to top](#table-of-contents)

---

## 3. Folder Map: What Is Where

```
CaPT/
├── README.md                 ← Short official instructions
├── LICENSE                   ← Legal terms for using the code
├── requirements.txt          ← List of Python packages to install
├── setup.py                  ← Installs the MODIFIED transformers library
├── train.sh                  ← Example commands to start training
│
├── capt/                     ← ★ TRAINING SCRIPTS (start here)
│   ├── train_t5.py           ← Train CaPT on T5 models
│   ├── train_llama.py        ← Train CaPT on Llama-3.2
│   └── train_qwen.py         ← Train CaPT on Qwen-2.5
│
├── src/                      ← DATA LOADING AND FORMATTING
│   ├── dataset.py            ← Turns text into token IDs (used by T5)
│   ├── superglue_loader_t5.py      ← Reads SuperGLUE files, builds text prompts
│   ├── superglue_loader_decoder.py ← Same, for Llama/Qwen
│   └── commonsenseqa_loader.py     ← Extra dataset (CommonsenseQA)
│
├── config/
│   └── model_config.py       ← Short names → full Hugging Face model names
│
├── superglue/
│   └── get_huggingface_superglue.py ← Downloads the 6 SuperGLUE datasets
│
├── images/
│   └── capt_overview.png     ← Architecture figure from the paper
│
└── transformers/             ← ★ A COPY of Hugging Face Transformers v4.43.2
    └── models/                 with CaPT added inside 3 model files:
        ├── t5/modeling_t5.py
        ├── llama/modeling_llama.py
        └── qwen2/modeling_qwen2.py
```

**Main point:** The `transformers/` folder is huge (about 250 model folders), but **only 3 files in it were changed for CaPT**. You can ignore the rest of it.

[Back to top](#table-of-contents)

---

## 4. What Each File Does

### 4.1 Top-level files

| File | What it does |
| --- | --- |
| [README.md](../CaPT/README.md) | Official short guide: install, download data, train, cite. |
| [requirements.txt](../CaPT/requirements.txt) | Exact versions of every package (PyTorch 2.4.0, datasets 2.20.0, etc.). The `transformers` line is commented out because the project uses its **own** modified copy. |
| [setup.py](../CaPT/setup.py) | Running `python setup.py install` installs the local `transformers/` folder as your `transformers` package, so `from transformers import ...` loads the CaPT version. |
| [train.sh](../CaPT/train.sh) | A shell script with 3 example commands (T5 on COPA, Llama on CB, Qwen on CB). |
| [images/capt_overview.png](../CaPT/images/capt_overview.png) | Figure 1 of the paper (Deep Prompt-Tuning vs. CaPT). |

### 4.2 The capt folder: training scripts

These are the files you actually **run**. All three follow the same pattern: *read settings → load data → build model → freeze → train → evaluate → save results*.

| File | Model type | How it predicts | Main parts |
| --- | --- | --- | --- |
| [train_t5.py](../CaPT/capt/train_t5.py) | T5 (encoder-decoder) | **Generates the answer as text**, for example the word `entailment` or `True`. Uses `model.generate()` and compares the generated word to the correct word. | `train()`, `train_epoch()`, `evaluate_epoch()`, `score()`, `prepare_dataset()` |
| [train_llama.py](../CaPT/capt/train_llama.py) | Llama-3.2 (decoder-only) | **Classification**: a small linear layer (`model.score`) outputs one score per class, and the highest one wins. | `LlamaClassificationDataset`, `pad_sequence_collate()`, `train_epoch()`, `evaluate_epoch()`, `train()` |
| [train_qwen.py](../CaPT/capt/train_qwen.py) | Qwen-2.5 (decoder-only) | Same as Llama (classification head). | Nearly identical to `train_llama.py`. |

**Important functions in `train_t5.py`:**
- `prepare_dataset()` ([train_t5.py:431](../CaPT/capt/train_t5.py#L431)): picks the right metric (accuracy, F1, …), builds train/dev/test `DataLoader`s, and pads each batch to the same length (`collate_fn`).
- `train()` ([train_t5.py:57](../CaPT/capt/train_t5.py#L57)): main function. Loads the model, freezes it, unfreezes CaPT, creates the optimizer, and runs epochs.
- `train_epoch()` ([train_t5.py:287](../CaPT/capt/train_t5.py#L287)): one pass over training data. Forward pass → loss → backward pass → optimizer step.
- `evaluate_epoch()` ([train_t5.py:360](../CaPT/capt/train_t5.py#L360)): runs the model on dev or test data without learning and collects predictions.
- `score()` ([train_t5.py:403](../CaPT/capt/train_t5.py#L403)): turns predictions into a number (accuracy, F1, or their average).

> Some code mentions `prompt-routing`, `consistency_alpha` or `gumbel`. These are leftovers from the **SMoP** project this code was built on. CaPT does not use them.

### 4.3 The src folder: data loaders

| File | What it does |
| --- | --- |
| [superglue_loader_t5.py](../CaPT/src/superglue_loader_t5.py) | Reads the downloaded `.pkl` files and turns each example into a text prompt plus an answer. Example for RTE: `"rte sentence1: <premise> sentence2: <hypothesis>"` → `"entailment"`. It also **creates the splits** (see the note below). Main entry: `get_superglue()`. |
| [superglue_loader_decoder.py](../CaPT/src/superglue_loader_decoder.py) | Same idea, used by the Llama/Qwen scripts. |
| [dataset.py](../CaPT/src/dataset.py) | `SuperGlueData` class (a PyTorch `Dataset`). Takes the text from the loader and **tokenizes** it (text → number IDs). The answer is tokenized too, because T5 generates text. Used by `train_t5.py`. |
| [commonsenseqa_loader.py](../CaPT/src/commonsenseqa_loader.py) | Loads an extra dataset, CommonsenseQA, directly from Hugging Face. It is not part of the paper's main table. |

**Note on splits:** SuperGLUE's official test set has no public answers. So the code does this:
- **T5 (text-to-text):** test = official *validation* set. dev = 10% taken out of the training set.
- **Other mode:** the official validation set is cut in half to make dev and test.

### 4.4 The config folder

| File | What it does |
| --- | --- |
| [model_config.py](../CaPT/config/model_config.py) | A dictionary that maps short names to Hugging Face names, for example `'t5-v1_1-base'` → `'google/t5-v1_1-base'`. If your name is not in the list, the script uses it as written. |

### 4.5 The superglue folder: dataset downloader

| File | What it does |
| --- | --- |
| [get_huggingface_superglue.py](../CaPT/superglue/get_huggingface_superglue.py) | Downloads **BoolQ, CB, COPA, MultiRC, RTE, WiC** from Hugging Face and saves them as `data/superglue/<name>/<split>.pkl` files. |

**The 6 tasks in simple words:**

| Dataset | Question type | Example answer | Metric |
| --- | --- | --- | --- |
| BoolQ | Yes/no question about a passage | `True` / `False` | Accuracy |
| CB | Does sentence A imply B? | `entailment` / `contradiction` / `neutral` | (Accuracy + F1) / 2 |
| COPA | Which choice is the more likely cause or effect? | `choice1` / `choice2` | Accuracy |
| MultiRC | Is this answer correct for the passage? | `True` / `False` | F1 |
| RTE | Does sentence A imply B? | `entailment` / `not_entailment` | Accuracy |
| WiC | Does the word mean the same thing in both sentences? | `True` / `False` | Accuracy |

### 4.6 The transformers folder: the modified library

This is a full copy of **Hugging Face Transformers 4.43.2** with CaPT added. Only these files matter:

| File | What was added |
| --- | --- |
| [models/t5/modeling_t5.py](../CaPT/transformers/models/t5/modeling_t5.py) | `PromptTuning` class (normal soft prompt, the baseline) at [line 646](../CaPT/transformers/models/t5/modeling_t5.py#L646). **`CaPT` class** at [line 664](../CaPT/transformers/models/t5/modeling_t5.py#L664). `T5Block` changed to create and insert the capsule ([line 698](../CaPT/transformers/models/t5/modeling_t5.py#L698)). `T5Stack` changed to mark the first and last layers ([line 1029](../CaPT/transformers/models/t5/modeling_t5.py#L1029)). |
| [models/llama/modeling_llama.py](../CaPT/transformers/models/llama/modeling_llama.py) | `PromptTuning` and **`CaPT`** classes at [line 828](../CaPT/transformers/models/llama/modeling_llama.py#L828) and [line 847](../CaPT/transformers/models/llama/modeling_llama.py#L847). `LlamaModel` creates one CaPT per layer and inserts it in `forward()` ([line 997](../CaPT/transformers/models/llama/modeling_llama.py#L997)). |
| [models/qwen2/modeling_qwen2.py](../CaPT/transformers/models/qwen2/modeling_qwen2.py) | Same changes as Llama, for Qwen2 (around [line 741](../CaPT/transformers/models/qwen2/modeling_qwen2.py#L741) to [line 912](../CaPT/transformers/models/qwen2/modeling_qwen2.py#L912)). |

### 4.7 Folders created when you run the code

These do **not** exist until you run something:

| Folder | Created by | Contains |
| --- | --- | --- |
| `data/superglue/` | the download script | Downloaded datasets (`.pkl` files) |
| `log/` | training scripts | TensorBoard logs (loss and score curves) |
| `results/` | training scripts | A `.txt` file with the final test score and training time |
| `t5_models/` | `train_t5.py` | Best model weights (`model_state_dict.pt`) |
| `models/` | training scripts | Folder for saved prompt models (Llama/Qwen saves here with `--save_model`) |

[Back to top](#table-of-contents)

---

## 5. Architecture: How CaPT Is Built

### 5.1 The CaPT module (the core code)

This is the actual CaPT code from [modeling_t5.py:664-696](../CaPT/transformers/models/t5/modeling_t5.py#L664-L696), with comments added:

```python
class CaPT(nn.Module):
    def __init__(self, config):
        self.prompt_len = config.prompt_len          # = 1 (one capsule)
        self.hidden_size = config.d_model            # = 768 for T5-Base
        # The ONLY trainable thing: one vector of 768 numbers  → "task-aware" part
        self.soft_prompts = nn.Parameter(torch.zeros(self.prompt_len, self.hidden_size))
        nn.init.xavier_normal_(self.soft_prompts)    # start with small random numbers

    def forward(self, hidden_states):                # hidden_states: [batch, tokens, 768]
        # Average over all tokens → one vector per sentence  → "instance-aware" part
        compressed_seq = torch.mean(hidden_states, dim=1, keepdim=True)   # [batch, 1, 768]
        # Copy the learned vector for every sentence in the batch
        soft = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 1, 768]
        return compressed_seq + soft                 # capsule = Mean(input) + p
```

**Picture:**

```
 Input tokens at this layer:   [ t1 ][ t2 ][ t3 ] ... [ tn ]
                                   \    |    |        /
                                    \   |    |       /
                                     Mean (average)        ← changes for every sentence
                                           |
                                           +  ←── p (learned vector, 768 numbers)  ← trainable
                                           |
                                      [ CAPSULE ]
                                           |
 New sequence for this layer:  [ CAPSULE ][ t1 ][ t2 ][ t3 ] ... [ tn ]
                                     ↑ placed FIRST, so it can act as the "attention anchor"
```

**Number of trainable parameters for T5-Base:**
`1 vector × 768 numbers × 12 encoder layers = 9,216 parameters`. This is the "≈9K" in the paper's figure. The full model has about 220 million.

### 5.2 How CaPT is plugged into T5

T5 has an **encoder** (reads the input) and a **decoder** (writes the answer). CaPT is added **only to the encoder layers** (`if not self.is_decoder`).

```
                     ENCODER (12 layers for T5-Base)                    DECODER (frozen, unchanged)
 Input text
    ↓ embedding
 [t1..tn]
    ↓
 Layer 0  (first_layer=True):  make capsule from [t1..tn]
                               CONCAT → [C][t1..tn]         (sequence gets 1 longer)
    ↓
 Layer 1..10:                  make new capsule from [C][t1..tn]
                               REPLACE position 0 → [C'][t1..tn]
    ↓
 Layer 11 (last_layer=True):   replace position 0, run layer,
                               then REMOVE capsule → [t1..tn]  ──────→  generates "entailment"
```

Details from [T5Block.forward](../CaPT/transformers/models/t5/modeling_t5.py#L752-L801):
- The **attention mask** gets one extra `1` at the front so the model is allowed to "see" the capsule.
- Layer 0 **adds** the capsule. Later layers **overwrite** it with a fresh one (paper formula `Sⁱ = pⁱ + Mean(Sⁱ⁻¹ ⊕ Hⁱ⁻¹)`). The last layer **removes** it so the decoder gets a normal-length input.
- `--prompt_layers all` creates the list of layers that get prompts ([T5Stack, line 1035](../CaPT/transformers/models/t5/modeling_t5.py#L1035)).

### 5.3 How CaPT is plugged into Llama and Qwen

These are decoder-only models, so the approach is slightly different ([LlamaModel.forward](../CaPT/transformers/models/llama/modeling_llama.py#L950-L1001)):

```
 Input text → embedding → [t1..tn]
                ↓
 Add one ZERO placeholder at the front → [0][t1..tn]   (attention mask also gets an extra 1)
                ↓
 For each of the 16 layers:
     capsule = CaPT_layer_i( Mean of current [x][t1..tn] )
     put capsule at position 0 → [C][t1..tn]
     run the frozen Llama layer
                ↓
 Remove position 0 → final hidden states
                ↓
 model.score (linear layer, TRAINABLE) → class scores → e.g. "neutral"
```

Differences from T5:
- There is one `CaPT` module per layer, stored in a list: `self.gen_prompt = nn.ModuleList([...])`.
- The input to the average is `hidden_states.detach()`. This means gradients do **not** flow back through the average. Only the learned vector `p` and the classification head learn.
- The **classification head** (`model.score`) is also trained. That is why the paper compares Llama/Qwen against "Linear Head" instead of full fine-tuning.

### 5.4 Paper equation to code mapping

| Paper | Meaning | Code |
| --- | --- | --- |
| `pⁱ` | Trainable task-aware vector for layer i | `self.soft_prompts` in `CaPT` |
| `E` | Input embeddings | `hidden_states` at layer 0 |
| `Hⁱ⁻¹` | Output of previous layer | `hidden_states` passed into each block |
| `Mean(·)` | Average over tokens | `torch.mean(sequence, dim=1, keepdim=True)` |
| `S¹ = p¹ + Mean(E)` | First capsule | Layer 0: `generated_prompt` concatenated in front |
| `Sⁱ = pⁱ + Mean(Sⁱ⁻¹ ⊕ Hⁱ⁻¹)` | Later capsules | Later layers: `hidden_states[:, :prompt_len, :] = generated_prompt` |
| Prompt length = 1 | One capsule | `--num_virtual_tokens 1` |
| Depth "all layers" | Capsule in every layer | `--prompt_layers all` (T5); always all layers for Llama/Qwen |
| Baseline Deep Prompt-Tuning | Learned vectors only, no input info | `PromptTuning` class, selected with `--generator pt` |

[Back to top](#table-of-contents)

---

## 6. Working Flow: What Happens When You Run It

```
┌──────────────────────────────┐
│ superglue/get_huggingface_   │  Step 1: download data → data/superglue/*.pkl
│ superglue.py                 │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ train.sh  OR  python         │  You start training with chosen options
│ capt/train_t5.py --...       │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ src/superglue_loader_*.py    │  Step 2: read .pkl, build text prompts,
│ src/dataset.py               │          split train/dev/test, tokenize
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ transformers/models/.../     │  Step 3: load pre-trained model;
│ modeling_*.py  (CaPT inside) │          CaPT modules are created inside each layer
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ train script                 │  Step 4: freeze model, unfreeze CaPT only
│                              │  Step 5: train for N epochs (Adafactor + warmup)
│                              │  Step 6: evaluate on dev each epoch, test when better,
│                              │          early stop, save model and results
└──────────────┬───────────────┘
               ↓
     log/   results/   t5_models/
```

### 6.1 Step 1: Download data
[get_huggingface_superglue.py](../CaPT/superglue/get_huggingface_superglue.py) calls `load_dataset('super_glue', name)` for each of the 6 tasks and saves each split with `pickle`.

### 6.2 Step 2: Load and format data
1. `prepare_dataset()` (T5) or `prepare_classification_datasets()` (Llama/Qwen) is called with the dataset name.
2. `get_superglue()` reads the `.pkl` files and makes `(input_text, answer)` pairs.
3. The text is tokenized into IDs. The input is cut at `max_length` tokens. For T5, `max_length` is reduced by the number of prompts to leave room for the capsule.
4. A `DataLoader` groups examples into batches and pads them to equal length.

### 6.3 Step 3: Build the model
```python
config = AutoConfig.from_pretrained("t5-base")
config.prompt_len = 1          # from --num_virtual_tokens
config.prompt_layers = "all"   # from --prompt_layers
config.generator = "capt"      # from --generator   (Llama/Qwen: config.gen_prompt)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base", config=config)
```
Because the modified `transformers` library is installed, building the model also creates the `CaPT` modules inside the layers. The pre-trained weights load as normal. The CaPT vectors start with small random values.

### 6.4 Step 4: Freeze everything except CaPT
```python
for param in model.parameters():
    param.requires_grad = False                 # freeze ALL
for block in model.encoder.block:
    for param in block.gen_prompt.parameters():
        param.requires_grad = True              # unfreeze only the capsule vectors
```
The script then prints the number of trainable parameters and the percentage (for example `0.004%`).

### 6.5 Step 5: Training loop
For each epoch, `train_epoch()` does this for every batch:
1. **Forward:** run the model and get the `loss`.
2. **Backward:** `scaler.scale(loss).backward()` computes how to change the capsule vectors.
3. **Update:** `scaler.step(optimizer)` changes the vectors. `scheduler.step()` adjusts the learning rate (it warms up first, then decreases linearly).
4. Write the loss to TensorBoard every 10 steps.

### 6.6 Step 6: Evaluate, early stop and save
After each epoch:
1. Run `evaluate_epoch()` on the **dev** set and compute the score.
2. **If the dev score (or loss) improved:** run on the **test** set, remember the test score, reset patience, and save the model.
3. **If it did not improve:** `patience -= 1`. When patience reaches 0, stop early.
4. At the end, print and save the best test score and the training time to `results/.../*.txt`.

The reported number is **the test score from the epoch with the best dev score**.

[Back to top](#table-of-contents)

---

## 7. Technology Used

| Technology | Version | What it is used for |
| --- | --- | --- |
| **Python** | 3.9.18 | Programming language |
| **PyTorch** (`torch`) | 2.4.0 | Deep learning framework: tensors, GPU math, automatic gradients |
| **CUDA** (NVIDIA libraries, `nvidia-*`) | 12.1 | Runs PyTorch on NVIDIA GPUs |
| **Hugging Face Transformers** (modified copy) | 4.43.2 | Pre-trained T5 / Llama / Qwen models and tokenizers, plus the CaPT changes |
| **Hugging Face Datasets** | 2.20.0 | Downloads SuperGLUE and CommonsenseQA |
| **Hugging Face Hub** | 0.24.2 | Downloads model weights |
| **Accelerate** | 0.33.0 | Helper for loading models on devices |
| **tokenizers** | 0.19.1 | Fast text → token ID conversion |
| **Adafactor** (from transformers) | – | Optimizer that uses little memory |
| **Mixed precision** (`torch.cuda.amp`: `GradScaler`, `autocast`) | – | Faster training with less memory |
| **tensorboardX** | 2.6.2.2 | Writes training curves you can view in TensorBoard |
| **scikit-learn** | 1.0.2 | F1 score and other metrics |
| **SciPy** | 1.13.1 | Spearman correlation metric |
| **NumPy / pandas** | 1.22.4 / 2.2.2 | Number and table handling |
| **tqdm** | 4.66.4 | Progress bars |
| **conda** | – | Creates an isolated Python environment |

**Pre-trained models used in the paper:**

| Model | Size | Type | Hugging Face name |
| --- | --- | --- | --- |
| T5-Base | 220M | Encoder-decoder | `t5-base` |
| T5-Large | 770M | Encoder-decoder | `t5-large` |
| Llama-3.2-1B | 1B | Decoder-only | `meta-llama/Llama-3.2-1B` (gated: needs access approval) |
| Qwen-2.5-1.5B | 1.5B | Decoder-only | `Qwen/Qwen2.5-1.5B` |

The code builds on the **SMoP** project ([github.com/jyjohnchoi/SMoP](https://github.com/jyjohnchoi/SMoP)).

[Back to top](#table-of-contents)

---

## 8. Running Requirements

### Hardware
| Item | Need |
| --- | --- |
| **GPU** | An **NVIDIA GPU with CUDA** is needed in practice. The paper used **NVIDIA RTX 6000 Ada (48 GB)**. T5-Base fits on smaller GPUs. Llama/Qwen and T5-Large need more memory. If you run out of memory, lower `--batch_size` or use `--step_batch_size`. |
| **CPU only** | The code falls back to CPU, but training would be extremely slow. Not recommended. |
| **Disk** | A few GB for datasets and models (T5-Base ≈ 0.9 GB, Llama-3.2-1B ≈ 2.5 GB, Qwen-2.5-1.5B ≈ 3 GB), plus saved checkpoints. |
| **Internet** | Needed the first time to download datasets and models. |

### Software
| Item | Need |
| --- | --- |
| **Operating system** | **Linux** is recommended. `requirements.txt` includes `triton==3.0.0` and `nvidia-*` packages, which are **Linux-only**. On Windows, use **WSL2 (Ubuntu)** (see [Common Problems](#11-common-problems-and-fixes)). |
| **conda** | Miniconda or Anaconda, to create the environment |
| **Python** | **3.9** exactly (the scripts use `distutils`, which was removed in Python 3.12) |
| **NVIDIA driver** | A recent driver that supports CUDA 12.1 |
| **Hugging Face account** | Needed only for Llama-3.2: request access on its model page, then run `huggingface-cli login` |

[Back to top](#table-of-contents)

---

## 9. How to Run: Step by Step

All commands must be run **from inside the `CaPT/` folder**, because the scripts use paths like `data/superglue/...` relative to where you are.

### Step 1: Create the environment
```bash
cd CaPT
conda create -n capt python=3.9.18 -y
conda activate capt
pip install -r requirements.txt
```

### Step 2: Install the modified transformers library
```bash
python setup.py install
```
Do **not** run `pip install transformers` afterwards. It would replace the CaPT version.

Quick check that the right version is installed:
```bash
python -c "from transformers.models.t5.modeling_t5 import CaPT; print('CaPT found')"
```

### Step 3: Download the datasets
```bash
python superglue/get_huggingface_superglue.py
```
> The README says `data/superglue/get_huggingface_superglue.py`, but in this folder the script is at `superglue/get_huggingface_superglue.py`. It saves files into `data/superglue/`.

### Step 4: Train (start small)
A good first run is **T5-Base on COPA**, because it is a small dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python capt/train_t5.py --method gen-prompt --generator capt \
  --model_name_or_path t5-base --tokenizer_name_or_path t5-base \
  --dataset_name copa --num_virtual_tokens 1 --prompt_layers all \
  --lr 1 --batch_size 32 --epoch 50 --max_length 512 \
  --warmup_ratio 0.01 --max_patience 20 --random_seed 88
```
Or run all three examples:
```bash
sh train.sh
```

**Llama** (needs Hugging Face login and access):
```bash
CUDA_VISIBLE_DEVICES=0 python capt/train_llama.py --generator capt --lr 7e-4 \
  --dataset_name cb --random_seed 88 --num_virtual_tokens 1 --epoch 50 \
  --batch_size 32 --warmup_ratio 0.03 --weight_decay 0.005 \
  --model_name_or_path meta-llama/Llama-3.2-1B
```

**Qwen:**
```bash
CUDA_VISIBLE_DEVICES=0 python capt/train_qwen.py --generator capt --lr 1e-1 \
  --dataset_name cb --random_seed 88 --num_virtual_tokens 1 --epoch 50 \
  --batch_size 16 --warmup_ratio 0.01 --weight_decay 0.005 --max_patience 20 \
  --model_name_or_path Qwen/Qwen2.5-1.5B
```

### Step 5: Read the results
- **Terminal:** each epoch prints train loss, validation loss and validation score. At the end it prints `Test result | Score : XX.XX`.
- **File:** `results/<model>/<dataset>/...txt`
- **Curves:** `tensorboard --logdir log` (install with `pip install tensorboard`), then open the link it prints in your browser.

**What you should see:** the trainable parameter count is printed near the start. For T5-Base with CaPT it should be about **9,216 parameters (≈0.004%)**.

[Back to top](#table-of-contents)

---

## 10. Command-Line Options Explained

| Option | Meaning | Typical value |
| --- | --- | --- |
| `--model_name_or_path` | Which pre-trained model to load | `t5-base`, `t5-large`, `meta-llama/Llama-3.2-1B`, `Qwen/Qwen2.5-1.5B` |
| `--tokenizer_name_or_path` | (T5 only) Which tokenizer to use; usually the same as the model | `t5-base` |
| `--method` | (T5 only) Must be `gen-prompt` to train the prompt modules | `gen-prompt` |
| `--generator` | Which prompt type: **`capt`** = this paper, `pt` = normal deep prompt-tuning baseline | `capt` |
| `--dataset_name` | Task to train on | `boolq`, `cb`, `copa`, `multirc`, `rte_superglue`, `wic` |
| `--num_virtual_tokens` | Number of capsule tokens. The paper's best is **1** | `1` |
| `--prompt_layers` | (T5 only) Which encoder layers get a capsule: `all` or a list like `0,2,4` | `all` |
| `--lr` | Learning rate. T5 uses large values like `1`, because Adafactor without relative step behaves differently from Adam | T5: `1`, Llama: `7e-4`, Qwen: `1e-1` |
| `--batch_size` | Examples per update | `16` or `32` |
| `--step_batch_size` | Split a batch into smaller pieces to save GPU memory (gradient accumulation) | e.g. `8` |
| `--epoch` | Maximum number of epochs | `50` |
| `--max_patience` | Epochs without improvement before early stop | `20` |
| `--max_length` | Maximum number of input tokens | `512` |
| `--warmup_ratio` | Fraction of training spent slowly raising the learning rate | `0.01`–`0.03` |
| `--weight_decay` | A small penalty that keeps weights from growing too large | `0.005` |
| `--random_seed` | Makes results repeatable | `88` |
| `--save_model` | (Llama only) Also save the best model with `save_pretrained` | flag |

**To reproduce paper experiments:**
- **Prompt length ablation (paper Fig. 6 top):** change `--num_virtual_tokens` to 2, 3, 5, 10, 20.
- **Depth ablation (paper Fig. 6 bottom, T5):** change `--prompt_layers`, for example `0` (input layer only), `0,1,2,3,4,5` (first half), or `1,3,5,7,9,11` (odd layers).
- **Baseline comparison:** `--generator pt`.

> The code also contains the pooling path (`AdaptiveAvgPool1d`) used when `prompt_len > 1`. The Prepending / Extraction / Projection variants from the paper's Table 3 are **not** in this code folder.

[Back to top](#table-of-contents)

---

## 11. Common Problems and Fixes

| Problem | Cause | Fix |
| --- | --- | --- |
| `pip install -r requirements.txt` fails on **Windows** (triton / nvidia packages) | These packages exist only for Linux | Use **WSL2 + Ubuntu** with NVIDIA's WSL CUDA driver, or a Linux machine / cloud GPU. |
| `sh train.sh` or `CUDA_VISIBLE_DEVICES=0 ...` does not work on Windows | That is Linux shell syntax | Run in WSL2 or Git Bash. In PowerShell, run `$env:CUDA_VISIBLE_DEVICES="0"` first, then the `python ...` command without the prefix. |
| `FileNotFoundError: data/superglue/...` | Data not downloaded, or you are not in the `CaPT/` folder | `cd CaPT`, then run `python superglue/get_huggingface_superglue.py`. |
| `Wrong dataset name 'rte'` | The code calls RTE `rte_superglue` | Use `--dataset_name rte_superglue`. |
| `ImportError: cannot import name 'CaPT'` or `AttributeError: ... 'prompt_len'` | The official `transformers` is being used instead of the modified one | `pip uninstall transformers -y`, then `python setup.py install` again inside `CaPT/`. |
| `ModuleNotFoundError: distutils` | Python 3.12 or newer | Use Python 3.9 (`conda create -n capt python=3.9.18`). |
| Llama download fails with 401/403 | Llama is a gated model | Accept the license on its Hugging Face page, then run `huggingface-cli login`. |
| Qwen trains with no prompts | `train_qwen.py` defaults to `--generator prompt-tuning`, which matches neither `capt` nor `pt` | Always pass `--generator capt`. |
| `CUDA out of memory` | GPU too small | Lower `--batch_size`, use `--step_batch_size`, lower `--max_length`, or start with `t5-base`. |
| Very slow training | Running on CPU | Check `python -c "import torch; print(torch.cuda.is_available())"`. It should print `True`. |

[Back to top](#table-of-contents)

---

## 12. Suggested Reading Order for the Code

If you want to understand the code yourself, read it in this order:

1. [train.sh](../CaPT/train.sh): see how training is started.
2. [train_t5.py → `train()`](../CaPT/capt/train_t5.py#L57): the main flow, from top to bottom.
3. [superglue_loader_t5.py → `rte()` and `get_superglue()`](../CaPT/src/superglue_loader_t5.py): how one dataset becomes text pairs.
4. [dataset.py → `SuperGlueData`](../CaPT/src/dataset.py): how text becomes token IDs.
5. [modeling_t5.py → `CaPT` class](../CaPT/transformers/models/t5/modeling_t5.py#L664): **the core idea**, about 25 lines.
6. [modeling_t5.py → `T5Block.forward`](../CaPT/transformers/models/t5/modeling_t5.py#L752): where the capsule is inserted and removed.
7. [modeling_llama.py → `LlamaModel.forward`](../CaPT/transformers/models/llama/modeling_llama.py#L950): the decoder-only version.
8. [train_llama.py](../CaPT/capt/train_llama.py): the classification-style training.

Keep the [paper summary](../capt-summary/All%20You%20Need%20is%20One%20Capsule%20Prompt%20Tuning%20with%20a%20Single%20vector.md) open next to the code, and use the [equation-to-code table](#54-paper-equation-to-code-mapping) to connect them.

[Back to top](#table-of-contents)
