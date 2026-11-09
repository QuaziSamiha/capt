# CaPT Datasets

How the CaPT code gets, splits, formats and scores its data.

> **Related docs:** [document.md](document.md) explains the code as a whole, and [running.md](running.md) explains how to run it.
> **Code folder:** [CaPT/](../CaPT/)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Supported Datasets](#2-supported-datasets)
3. [Downloading the Data](#3-downloading-the-data)
4. [On-Disk Layout](#4-on-disk-layout)
5. [Train / Dev / Test Splits](#5-train--dev--test-splits)
6. [How Each Task Is Turned Into Text](#6-how-each-task-is-turned-into-text)
7. [The Data Pipeline, File by File](#7-the-data-pipeline-file-by-file)
8. [Batches: What the Model Receives](#8-batches-what-the-model-receives)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Adding a New Dataset](#10-adding-a-new-dataset)
11. [Known Quirks and Gotchas](#11-known-quirks-and-gotchas)

---

## 1. Overview

CaPT is tested on **natural language understanding (NLU)** tasks. Every task is a
**classification problem**: the model reads some text and picks one answer from a
small, fixed set of labels.

The code supports:

- **6 tasks from the SuperGLUE benchmark:** BoolQ, CB, COPA, MultiRC, RTE and WiC.
- **1 extra task:** CommonsenseQA.

All data comes from the **Hugging Face Hub** through the `datasets` library.

The same raw data is used by every model, but it reaches each model differently:

| Model family | Training script | Loader used | How the model answers |
|---|---|---|---|
| T5 (encoder-decoder) | [capt/train_t5.py](../CaPT/capt/train_t5.py) | [src/dataset.py](../CaPT/src/dataset.py) → [src/superglue_loader_t5.py](../CaPT/src/superglue_loader_t5.py) | It **writes** the label as text, for example `entailment` |
| Llama / Qwen (decoder-only) | [capt/train_llama.py](../CaPT/capt/train_llama.py), [capt/train_qwen.py](../CaPT/capt/train_qwen.py) | `LlamaClassificationDataset` → [src/superglue_loader_decoder.py](../CaPT/src/superglue_loader_decoder.py) | A classification head **picks a label number** (0, 1, 2…) |

---

## 2. Supported Datasets

Use the name in the **`--dataset_name`** column when you run a training script.

| `--dataset_name` | Task | What the model must do | Labels | Metric |
|---|---|---|---|---|
| `boolq` | BoolQ (Boolean Questions) | Read a passage and answer a yes/no question about it | `False`, `True` | Accuracy |
| `cb` | CommitmentBank | Decide whether a premise entails, contradicts or is neutral toward a hypothesis | `entailment`, `contradiction`, `neutral` | Mean of accuracy and macro-F1 |
| `copa` | Choice of Plausible Alternatives | Given a premise, pick the more likely cause or effect of two choices | `choice1`, `choice2` | Accuracy |
| `multirc` | Multi-Sentence Reading Comprehension | Given a paragraph, a question and one candidate answer, decide whether the answer is correct | `False`, `True` | Macro-F1 (`f1a`) |
| `rte_superglue` | Recognizing Textual Entailment | Decide whether a premise entails a hypothesis | `entailment`, `not_entailment` | Accuracy |
| `wic` | Word-in-Context | Decide whether a word means the same thing in two sentences | `False`, `True` | Accuracy |
| `commonsense_qa` | CommonsenseQA | Answer a common-sense question by picking one of five options | `A`, `B`, `C`, `D`, `E` | Accuracy |

> **Note:** RTE is passed as **`rte_superglue`**, not `rte`. The plain name `rte` is rejected.

### Approximate sizes (official Hugging Face splits)

| Dataset | Official train | Official validation | Used as test here |
|---|---:|---:|---|
| BoolQ | 9,427 | 3,270 | official validation |
| CB | 250 | 56 | official validation |
| COPA | 400 | 100 | official validation |
| MultiRC | 27,243 | 4,848 | official validation |
| RTE | 2,490 | 277 | official validation |
| WiC | 5,428 | 638 | official validation |
| CommonsenseQA | 9,741 | 1,221 | official validation |

MultiRC sizes count **(question, answer) pairs**, not paragraphs: each candidate answer is its own example.

CB and COPA are very small. Results on them change a lot from one random seed to the next.

---

## 3. Downloading the Data

### SuperGLUE tasks

Run the downloader **from inside the `CaPT/` folder**:

```bash
cd CaPT
python superglue/get_huggingface_superglue.py
```

> The [README](../CaPT/README.md) says `python data/superglue/get_huggingface_superglue.py`. That path is wrong. The script is at [superglue/get_huggingface_superglue.py](../CaPT/superglue/get_huggingface_superglue.py).

The script ([source](../CaPT/superglue/get_huggingface_superglue.py)):

1. Loops over `['boolq', 'cb', 'copa', 'multirc', 'rte', 'wic']`.
2. Downloads each one with `load_dataset('super_glue', <name>)`.
3. Saves every split (`train`, `validation`, `test`) as a **pickle file** under `data/superglue/<name>/`.

The loaders open `data/superglue/...` as a **relative path**. You must therefore start training from the `CaPT/` folder too, or the loaders will not find the files.

### CommonsenseQA

There is **nothing to download in advance**. The loader calls `load_dataset("tau/commonsense_qa")` every time training starts. Hugging Face caches it after the first run.

If that call fails (for example, when there is no internet), the loader looks for these local files instead:

```
data/commonsenseqa/train_rand_split.jsonl
data/commonsenseqa/dev_rand_split.jsonl
```

These are the original CommonsenseQA release files. Each line is one JSON object with `id`, `question`, `choices` (`label` and `text` lists) and `answerKey`.

---

## 4. On-Disk Layout

After the downloader runs, the folder looks like this:

```
CaPT/
└── data/
    └── superglue/
        ├── boolq/    train.pkl  validation.pkl  test.pkl
        ├── cb/       train.pkl  validation.pkl  test.pkl
        ├── copa/     train.pkl  validation.pkl  test.pkl
        ├── multirc/  train.pkl  validation.pkl  test.pkl
        ├── rte/      train.pkl  validation.pkl  test.pkl
        └── wic/      train.pkl  validation.pkl  test.pkl
```

Each `.pkl` file holds a pickled Hugging Face `datasets.Dataset` object. The loaders read only `train.pkl` and `validation.pkl`. **`test.pkl` is never used**, because SuperGLUE test labels are hidden (every label is `-1`).

### Raw fields per task

| Task | Fields used by the loaders |
|---|---|
| BoolQ | `question`, `passage`, `label` |
| CB | `premise`, `hypothesis`, `label` |
| COPA | `premise`, `choice1`, `choice2`, `question` (`cause` / `effect`), `label` |
| MultiRC | `paragraph`, `question`, `answer`, `label` |
| RTE | `premise`, `hypothesis`, `label` |
| WiC | `sentence1`, `sentence2`, `word`, `label` |
| CommonsenseQA | `id`, `question`, `choices.label`, `choices.text`, `answerKey` |

---

## 5. Train / Dev / Test Splits

The official test labels are hidden, so the code builds its **own three splits**:

```
            Official train                      Official validation
   ┌──────────────────────────────┐          ┌───────────────────┐
   │        90%         │  10%    │          │       100%        │
   └──────────────────────────────┘          └───────────────────┘
            │               │                          │
          train            dev                        test
   (used for learning) (picks best epoch)   (final reported score)
```

All training scripts use this **text-to-text mode** (`text_to_text=True`):

- **test** = the full official validation set.
- **train / dev** = the official train set split 90% / 10%.
  - For the **SuperGLUE tasks**, the split is **stratified by label**. Examples are grouped by label, and the first 90% of each group goes to train while the last 10% goes to dev. Every label keeps the same share in both splits. See `construct_dev_from_train_t2t` in [src/superglue_loader_t5.py:19](../CaPT/src/superglue_loader_t5.py#L19).
  - For **CommonsenseQA**, the split is a plain cut: the first 90% of rows go to train and the rest to dev ([src/commonsenseqa_loader.py:7](../CaPT/src/commonsenseqa_loader.py#L7)).

The split is **deterministic**: the code does not shuffle before cutting, so every run gets the same train/dev/test sets. The random seed changes only the training order.

> **Legacy mode (`text_to_text=False`).** In this mode, the official train set stays whole and the official validation set is cut in half: the first half becomes dev and the second half becomes test ([src/superglue_loader_t5.py:41](../CaPT/src/superglue_loader_t5.py#L41)). No script in this repo uses this mode. It is left over from the [SMoP](https://github.com/jyjohnchoi/SMoP) code base.

---

## 6. How Each Task Is Turned Into Text

Every raw example becomes a pair: **(input text, label)**. The functions `boolq()`, `cb()`, `copa()`, `multirc()`, `rte()` and `wic()` in the loader files do this conversion, along with `commonsenseqa()` for CommonsenseQA.

### 6.1 Text-to-text format (used by all models)

| Task | Input text | Target label text |
|---|---|---|
| BoolQ | `boolq passage: {passage} question: {question}` | `False` / `True` |
| CB | `cb hyopthesis: {hypothesis}. premise: {premise}` | `entailment` / `contradiction` / `neutral` |
| COPA | `choice1: {choice1} choice2: {choice2} premise: {premise} question: {question}` | `choice1` / `choice2` |
| MultiRC | `question: {question} answer: {answer}. paragraph: {paragraph}` | `False` / `True` |
| RTE | `rte sentence1: {premise} sentence2: {hypothesis}` | `entailment` / `not_entailment` |
| WiC | `wic sentence1: {sentence1} sentence2: {sentence2} word: {word}` | `False` / `True` |
| CommonsenseQA | `commonsenseqa question: {question} choices: A: {a}, B: {b}, C: {c}, D: {d}, E: {e}` | `A` … `E` |

These prefixes follow the original T5 style. The spelling `hyopthesis` in CB is a typo in the code, but the model learns it anyway, so it causes no problems. COPA and MultiRC have **no task prefix**. Their `text_format` variable is defined but never used.

**Example (CB):**

```
Input : cb hyopthesis: the kids were hungry. premise: The children asked for food again.
Target: entailment
```

### 6.2 How T5 uses it

[src/dataset.py](../CaPT/src/dataset.py) (`SuperGlueData.formatting`) does the following:

1. It tokenizes the input text and adds T5's special tokens (`</s>` at the end).
2. It cuts the input to `--max_length` tokens (512 by default).
3. It tokenizes the **label text** plus the EOS token, for example `entailment</s>`. This becomes the decoder target.

During evaluation, T5 **generates** text. The generated text is decoded and compared, as a string, to the decoded target.

### 6.3 How Llama / Qwen use it

`LlamaClassificationDataset` in [capt/train_llama.py:103](../CaPT/capt/train_llama.py#L103) (the Qwen script has an identical copy) does the following:

1. It loads the same text-to-text pairs from [src/superglue_loader_decoder.py](../CaPT/src/superglue_loader_decoder.py).
2. It turns each **label string into an integer** using the `label_map` in `TASK_CONFIGS` ([capt/train_llama.py:32](../CaPT/capt/train_llama.py#L32)):

   | Task | Label → index |
   |---|---|
   | `rte_superglue` | `not_entailment` → 0, `entailment` → 1 |
   | `cb` | `entailment` → 0, `contradiction` → 1, `neutral` → 2 |
   | `boolq`, `wic`, `multirc` | `False`/`No` → 0, `True`/`Yes` → 1 |
   | `copa` | `choice1` → 0, `choice2` → 1 |
   | `commonsense_qa` | `A` → 0 … `E` → 4 |

3. It tokenizes the input with truncation to `max_length`, without padding.
4. A sequence-classification head (`LlamaForSequenceClassification` / `Qwen2ForSequenceClassification`) predicts the index.

> **Important:** `TASK_CONFIGS` also defines readable prompt templates such as `"Premise: …\nHypothesis: …\nAnswer:"`. **In practice these templates are not used.** `format_input()` applies a template only when the input is a tuple. In text-to-text mode, every input is already a plain string, so the code falls through to `return str(input_seq)`. The decoder models therefore see the same text as T5 (section 6.1). If you want the readable templates, you have to change `format_input()`.

---

## 7. The Data Pipeline, File by File

```
 Hugging Face Hub
      │  superglue/get_huggingface_superglue.py      (one-time download)
      ▼
 data/superglue/<task>/{train,validation}.pkl
      │
      │  src/superglue_loader_t5.py / src/superglue_loader_decoder.py
      │    <task>(text_to_text)     → raw rows → (text, label) pairs
      │    get_superglue(name, split) → builds train / dev / test
      │  src/commonsenseqa_loader.py  (CommonsenseQA, loaded straight from HF)
      ▼
 list of (input_text, label)
      │
      ├── T5:          src/dataset.py · SuperGlueData        → token ids + label token ids
      └── Llama/Qwen:  train_llama.py · LlamaClassificationDataset → token ids + label index
      ▼
 collate function  → padded batch tensors
      ▼
 torch DataLoader  → training loop
```

| File | Role |
|---|---|
| [superglue/get_huggingface_superglue.py](../CaPT/superglue/get_huggingface_superglue.py) | One-time download of the 6 SuperGLUE tasks to pickle files. |
| [src/superglue_loader_t5.py](../CaPT/src/superglue_loader_t5.py) | One function per task that turns raw rows into `(text, label)`. Also holds `get_superglue()`, the entry point that builds the splits and sends `commonsense_qa` to its own loader. |
| [src/superglue_loader_decoder.py](../CaPT/src/superglue_loader_decoder.py) | A near-identical copy of the T5 loader. Only comments and blank lines differ. The Llama and Qwen scripts import it. |
| [src/commonsenseqa_loader.py](../CaPT/src/commonsenseqa_loader.py) | Loads CommonsenseQA from Hugging Face, with a local-file fallback, and builds its splits. |
| [src/dataset.py](../CaPT/src/dataset.py) | `SuperGlueData`, the PyTorch `Dataset` for T5. It tokenizes and shuffles. `create_dataset(debug=True)` keeps only the first 64 examples, which is handy for quick tests. |
| [capt/train_t5.py](../CaPT/capt/train_t5.py) | `prepare_dataset()` ([line 431](../CaPT/capt/train_t5.py#L431)) builds the three datasets, picks the metric and wraps everything in DataLoaders. |
| [capt/train_llama.py](../CaPT/capt/train_llama.py) / [capt/train_qwen.py](../CaPT/capt/train_qwen.py) | `TASK_CONFIGS`, `LlamaClassificationDataset`, `pad_sequence_collate()` and `prepare_classification_datasets()` ([line 307](../CaPT/capt/train_llama.py#L307)). |

### The `get_superglue()` entry point

```python
from src.superglue_loader_t5 import get_superglue

data_list, num_labels = get_superglue("cb", split="train", text_to_text=True)
# data_list -> [("cb hyopthesis: ... premise: ...", "entailment"), ...]
# num_labels -> 3
```

- `split` must be `"train"`, `"dev"` or `"test"`.
- It returns a list of `(input, label)` tuples and the number of classes.

---

## 8. Batches: What the Model Receives

### T5 batch (`collate_fn` in `prepare_dataset`)

| Key | Shape | Content |
|---|---|---|
| `ids` | `[B]` | Example index |
| `input_ids` | `[B, L]` | Input tokens, right-padded with `pad_token_id` |
| `segment_ids` | `[B, L]` | Token type ids (all zeros for T5) |
| `label` | `[B, T]` | Target label tokens, padded with **`-100`** so the loss ignores the padding |

The training loop builds the attention mask itself as `input_ids != pad_token_id`.

### Llama / Qwen batch (`pad_sequence_collate`)

| Key | Shape | Content |
|---|---|---|
| `input_ids` | `[B, L]` | Input tokens, right-padded with `pad_token_id` (set to EOS if the tokenizer has no pad token) |
| `attention_mask` | `[B, L]` | 1 for real tokens, 0 for padding |
| `label` | `[B]` | Integer class index |

`L` is the longest sequence **in that batch**, so batches have different lengths.

### DataLoader settings

- The **train** loader uses `shuffle=True`. The **dev** and **test** loaders use `shuffle=False`.
- A seeded `torch.Generator` and `seed_worker` make the order repeatable with `--random_seed`.
- The T5 loaders use `num_workers=4` and `pin_memory=True`. On Windows or CPU-only machines, you may need to lower `num_workers` to 0.

---

## 9. Evaluation Metrics

The `score()` function in each training script computes the metrics ([capt/train_t5.py:403](../CaPT/capt/train_t5.py#L403), [capt/train_llama.py:459](../CaPT/capt/train_llama.py#L459)).

| Metric key | Used for | How it is computed |
|---|---|---|
| `acc` | BoolQ, COPA, RTE, WiC, CommonsenseQA | Share of predictions that exactly match the answer |
| `acc_f1_avg` | CB | `(accuracy + macro-F1) / 2` |
| `f1a` | MultiRC | Macro-F1 over all (question, answer) pairs |

Notes:

- For T5, predictions and answers are **decoded strings**. If T5 generates any text other than a valid label, the prediction counts as wrong.
- The `f1a` here is **sklearn macro-F1**, not the official SuperGLUE F1a (binary F1 on the positive class). The code also does not report MultiRC exact match (EM). Scores may differ slightly from the official leaderboard numbers.
- The model is scored on **dev** after each epoch. The **test** score is recorded at the epoch with the best dev score. Training stops early after `--max_patience` epochs without improvement on dev.
- Results are written to `results/<model>/<dataset>/`.

---

## 10. Adding a New Dataset

To add a new classification task, for example `mytask`:

1. **Get the data.** Save `train.pkl` and `validation.pkl` to `data/superglue/mytask/`, or load the data straight from Hugging Face the way [src/commonsenseqa_loader.py](../CaPT/src/commonsenseqa_loader.py) does.
2. **Write a loader function** in **both** [src/superglue_loader_t5.py](../CaPT/src/superglue_loader_t5.py) and [src/superglue_loader_decoder.py](../CaPT/src/superglue_loader_decoder.py):

   ```python
   def mytask(text_to_text):
       path = "data/superglue/mytask"
       train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
       val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))
       labels = ("negative", "positive")
       fmt = "mytask sentence: **text**"
       train_list = [(fmt.replace("**text**", d['text']), labels[d['label']]) for d in train]
       val_list   = [(fmt.replace("**text**", d['text']), labels[d['label']]) for d in val]
       return train_list, val_list, len(labels)
   ```

3. **Register it** in the `data_funcs` dict inside `get_superglue()` in both loader files.
4. **For T5:** add the name to `superglue_dataset_names` in `prepare_dataset()` ([capt/train_t5.py:431](../CaPT/capt/train_t5.py#L431)). If the task needs a metric other than accuracy, add it to `dataset_metrics` too.
5. **For Llama/Qwen:** add an entry to `TASK_CONFIGS` in **both** [capt/train_llama.py](../CaPT/capt/train_llama.py) and [capt/train_qwen.py](../CaPT/capt/train_qwen.py). It needs a `label_map` from every label string to an index, plus `orig_labels`. Set the metric in `LlamaClassificationDataset.__init__` if it is not accuracy.

---

## 11. Known Quirks and Gotchas

| Issue | Where | Effect |
|---|---|---|
| The README gives the downloader path as `data/superglue/...` | [README.md](../CaPT/README.md) | The command fails. Use `python superglue/get_huggingface_superglue.py`. |
| Data paths are relative (`data/superglue/...`) | All loaders | You must run the scripts from the `CaPT/` folder. |
| RTE's name is `rte_superglue` | `get_superglue()` | `--dataset_name rte` raises an error. The data folder is still `data/superglue/rte/`. |
| `semeval` appears in the T5 list of supported datasets | [capt/train_t5.py:432](../CaPT/capt/train_t5.py#L432) | It is not implemented, and `get_superglue()` rejects it. |
| ReCoRD and WSC appear in a comment in `SuperGlueData` | [src/dataset.py:19](../CaPT/src/dataset.py#L19) | They are not supported or downloaded. |
| The Llama/Qwen prompt templates are never applied | `format_input()` | The decoder models see T5-style inputs (see section 6.3). |
| CommonsenseQA has no `test` split when `text_to_text=False` | [src/commonsenseqa_loader.py:109](../CaPT/src/commonsenseqa_loader.py#L109) | Asking for `split="test"` crashes with an `UnboundLocalError`. This mode is not used by any script. |
| `hyopthesis` and `beacuse` typos | CB and COPA loaders | These are cosmetic. The model learns the text as written. |
| T5 truncation cuts only `input_ids` | `SuperGlueData.formatting` | `segment_ids` can be longer than `input_ids` for very long inputs. T5 ignores `segment_ids`, so training is not affected. |
| CommonsenseQA is downloaded at every start-up | `commonsenseqa()` | The first run needs internet. Later runs use the Hugging Face cache. |
