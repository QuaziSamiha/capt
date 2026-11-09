# Python Topics to Learn for the CaPT Codebase

A study list of the Python topics (and Python libraries) you need to read and understand the CaPT code.
Topics are in the order you should learn them. Each one says **where it is used** so you can practise by reading real code.

> **Tip:** You do **not** need to read the [CaPT/transformers/](../CaPT/transformers/) folder. It is a copy of the Hugging Face library.
> The only parts the authors changed are the `CaPT` / `PromptTuning` classes and the `forward()` method of the models in
> [modeling_llama.py](../CaPT/transformers/models/llama/modeling_llama.py), [modeling_qwen2.py](../CaPT/transformers/models/qwen2/modeling_qwen2.py) and [modeling_t5.py](../CaPT/transformers/models/t5/modeling_t5.py).
> Focus on [capt/](../CaPT/capt/), [src/](../CaPT/src/), [config/](../CaPT/config/) and [superglue/](../CaPT/superglue/).

---

## Level 1: Core Python Basics

- [ ] **Variables and basic types:** `int`, `float`, `str`, `bool`, `None`
- [ ] **Operators:** `//` (integer division), `%` (modulo), `**` (power, e.g. `2**32`), `^` (XOR on booleans)
  - Used in: [superglue_loader_decoder.py](../CaPT/src/superglue_loader_decoder.py) (`len(train_list) * 8 // 9`), the training loops (`a % accum_steps == 0`)
- [ ] **`if` / `elif` / `else`**, comparisons, `in` / `not in`
- [ ] **`for` loops**, `range()`, `break`, `continue`
  - Used in: the epoch loop and early stopping in [train_llama.py](../CaPT/capt/train_llama.py)
- [ ] **Functions:** parameters, default values (`max_length=512`), keyword arguments, returning **multiple values** as a tuple
  - Example: `prepare_classification_datasets()` returns 6 values at once
- [ ] **`print()`** and basic debugging

## Level 2: Data Structures

- [ ] **Lists:** indexing, `append`, `extend`, **slicing** (`list[:border]`, `list[border:]`)
- [ ] **Tuples** and **tuple unpacking**, including **nested unpacking**
  - Example: `((premise_joiner, choice1), (_, choice2)) = input_seq` in [train_llama.py](../CaPT/capt/train_llama.py)
  - The data loaders return lists of `(input, label)` tuples, sometimes `((text_a, text_b), label)`
- [ ] **Dictionaries:** nested dicts, `.keys()`, `.items()`, looking up keys
  - Example: `TASK_CONFIGS` in [train_llama.py](../CaPT/capt/train_llama.py) stores the prompt template and labels for each task
- [ ] **A dict of functions** (a "dispatch table")
  - Example: `data_funcs = {'boolq': boolq, 'cb': cb, ...}` in `get_superglue()`
- [ ] **Sets:** `set(string.punctuation)`
- [ ] **`collections.defaultdict`**: used to group examples by label in `construct_dev_from_train_t2t()`

## Level 3: Writing "Pythonic" Code

- [ ] **List comprehensions**: used everywhere in [superglue_loader_decoder.py](../CaPT/src/superglue_loader_decoder.py)
  - `[(text, labels[d['label']]) for d in train]`
- [ ] **Dict comprehensions**: `{v: k for k, v in self.label_map.items()}` (flips a dictionary)
- [ ] **Generator expressions** inside `sum()` / `max()`
  - `sum(p.numel() for p in model.parameters() if p.requires_grad)` counts trainable parameters
- [ ] **`enumerate()`** and **`zip()`**
  - `sum(a == b for a, b in zip(preds, answers))` computes accuracy
- [ ] **Conditional (ternary) expressions:** `x if cond else y`
- [ ] **`lambda` functions**: `collate_fn=lambda batch: pad_sequence_collate(batch, tokenizer)`
- [ ] **Nested functions (closures)**: `normalize_answer()` and `seed_worker()` are defined inside other functions

## Level 4: Strings and Text Processing

The data loaders spend most of their code turning raw data into text prompts.

- [ ] **f-strings** and **format specifiers**
  - `f"{val_score*100:.2f}%"` (2 decimals), `f"{trainable_params:,}"` (thousands separator), `f"{x:.4%}"` (percent)
- [ ] **Self-documenting f-strings** (Python 3.8+): `f"{lr=} {batch_size=}"` prints `lr=0.1 batch_size=16`
  - This is how the code builds log folder names and result-file lines
- [ ] **`str.format()` with named placeholders**: `"Premise: {premise}".format(premise=text_a)`
- [ ] **String methods:** `split`, `join`, `strip`, `replace`, `lower`
- [ ] **Multi-line strings and `\n`**: used in the prompt templates
- [ ] **Regular expressions (`re` module):** `re.sub`, `re.search`, `match.group(1)`, raw strings `r"..."`, `\b` and `\w+`
  - Used in: `normalize_answer()` and WiC word extraction

## Level 5: Error Handling and Type Checks

- [ ] **`raise ValueError(...)` / `raise TypeError(...)`**: input checks in the dataset classes
- [ ] **`try` / `except`**, including `except KeyError` (in [train_t5.py](../CaPT/capt/train_t5.py)) and bare `except:` (and why it is risky)
- [ ] **`isinstance()`** and **`type()`**: the code checks whether an input is a `str`, `tuple`, or `int`
- [ ] **`hasattr()`** and **`getattr(obj, name, default)`**
  - `getattr(config, 'compress_to', self.prompt_len)` in the `CaPT` class

## Level 6: Object-Oriented Programming (very important)

Every model and dataset in this project is a class.

- [ ] **Classes, `__init__`, `self`, instance attributes**
  - Simple example: `Config` in [model_config.py](../CaPT/config/model_config.py)
- [ ] **Methods** and calling one method from another (`self.create_dataset()`)
- [ ] **Inheritance** and **`super().__init__()`**
  - `class LlamaClassificationDataset(Dataset)`, `class CaPT(nn.Module)`
- [ ] **Magic / dunder methods:** `__len__` and `__getitem__` (needed for PyTorch datasets), `__call__` (why `model(x)` runs `forward()`)
- [ ] **Overriding methods**: `forward()` in `nn.Module` subclasses
- [ ] **Private-by-convention methods**: `_compress_sequence()` (leading underscore)
- [ ] **Decorators** (just recognise them): `@add_start_docstrings(...)` in the model files

## Level 7: Modules, Packages and Scripts

- [ ] **`import`, `from ... import ...`**, and **relative imports** (`from .commonsenseqa_loader import ...`)
- [ ] **`sys.path.append(os.getcwd())`**: why the scripts must be run from the [CaPT/](../CaPT/) folder
- [ ] **`if __name__ == "__main__":`**: the entry point of each training script
- [ ] **Packages and `setup.py`** (`setuptools.setup`, `find_packages`): [setup.py](../CaPT/setup.py) installs the modified `transformers` library in place of the official one
- [ ] **Virtual environments** (`conda`), **`pip install -r requirements.txt`**, pinned versions
- [ ] **Python version:** the project targets **Python 3.9**. `distutils` (used in `train_llama.py`) was removed in Python 3.12

## Level 8: Standard Library Modules Used

| Module | What it is used for | Where |
|---|---|---|
| `argparse` | Command-line options (`--lr`, `--dataset_name`, `action='store_true'`) | bottom of every `train_*.py` |
| `os` | `os.environ`, `os.makedirs(..., exist_ok=True)`, `os.path.join`, `os.path.basename` | training scripts |
| `sys` | `sys.path` | all scripts |
| `pickle` | Save/load the downloaded datasets as `.pkl` files | [get_huggingface_superglue.py](../CaPT/superglue/get_huggingface_superglue.py), loaders |
| `random` | Seeding for reproducibility | `seed_everything()` |
| `time` | Measuring training time | `train_epoch()` |
| `re`, `string` | Text cleaning | `normalize_answer()` |
| `collections` | `defaultdict` | loaders |
| `typing` | Type hints: `Optional`, `Union`, `Tuple`, `List` | model files |

- [ ] **File I/O with `with open(path, 'a') as f:`**: append mode, used to write results
- [ ] **Context managers (`with`)** in general: also used for `torch.no_grad()` and `autocast()`

## Level 9: NumPy Basics

- [ ] Arrays, shapes, `np.random.seed()`
- [ ] Understand **"shape thinking"** (rows × columns). This is the most important idea for PyTorch next.

## Level 10: PyTorch (the main library)

### Tensors
- [ ] Creating tensors: `torch.tensor`, `torch.zeros`, `torch.ones`, `torch.full`, `torch.randn`, `torch.LongTensor`
- [ ] **Shapes and dimensions**: `.shape`, `.size(0)`, `dim=`, `keepdim=True`
  - The key shape here is `(batch_size, seq_len, hidden_size)`
- [ ] Combining tensors: `torch.cat` (join along an existing dim) vs `torch.stack` (new dim)
- [ ] Reshaping: `unsqueeze`, `squeeze`, `expand`, `transpose`, `.contiguous()`
- [ ] **Tensor slicing**: `hidden_states[:, :self.prompt_len, :]` and `logits[..., prompt_length:-1, :]` (`...` = Ellipsis)
- [ ] Reductions: `torch.mean(x, dim=1, keepdim=True)` (this **is** the CaPT "instance-aware" compression), `torch.argmax`
- [ ] Moving data: `.to(device)`, `.cpu()`, `.item()`, `.tolist()`, `torch.cuda.is_available()`
- [ ] Data types: `dtype=torch.long`, `.float()`, `.long()`

### Building models
- [ ] **`nn.Module`**: `__init__` defines layers, `forward()` defines computation
- [ ] **`nn.Parameter`**: a learnable tensor (the CaPT soft prompt)
- [ ] **`nn.ModuleList`**: one CaPT module per transformer layer
- [ ] **`nn.init.xavier_normal_`**: weight initialisation
- [ ] **`nn.AdaptiveAvgPool1d`**: pooling when the prompt length is more than 1
- [ ] **`.detach()`**: stops gradients flowing back (used before feeding hidden states into CaPT)
- [ ] `model.parameters()`, `model.named_parameters()`, `p.numel()`

### Training
- [ ] **Freezing weights** with `param.requires_grad = False`, then unfreezing only the CaPT parameters and the classifier head
- [ ] **The training loop:** forward pass → loss → `loss.backward()` → `optimizer.step()` → `optimizer.zero_grad()`
- [ ] **`model.train()` vs `model.eval()`**, and **`torch.no_grad()`** for evaluation
- [ ] **Gradient accumulation** (`accum_steps`)
- [ ] **Mixed precision:** `torch.cuda.amp.autocast` and `GradScaler` (`scaler.scale(loss).backward()`, `scaler.step()`, `scaler.update()`)
- [ ] **Learning-rate schedulers** (`scheduler.step()`) and **warmup**
- [ ] **Early stopping** with a patience counter (plain Python logic in `train()`)

### Data loading
- [ ] **`torch.utils.data.Dataset`**: implement `__len__` and `__getitem__`
- [ ] **`DataLoader`**: `batch_size`, `shuffle`, `num_workers`
- [ ] **Custom `collate_fn`**: `pad_sequence_collate()` pads each batch to its longest sequence
- [ ] **Reproducibility:** `worker_init_fn`, `torch.Generator().manual_seed()`, `torch.initial_seed()`

## Level 11: Hugging Face Libraries

- [ ] **`transformers`**
  - `AutoTokenizer.from_pretrained(...)`: tokenising text, `input_ids`, `attention_mask`, `padding`, `truncation`, `return_tensors="pt"`
  - Special tokens: `pad_token`, `eos_token`, and why Llama needs `pad_token = eos_token`
  - `AutoConfig`: the code **adds custom fields** to the config (`config.prompt_len`, `config.gen_prompt`) and the model reads them in `__init__`
  - `AutoModelForCausalLM`, `AutoModelForSeq2SeqLM`, `LlamaForSequenceClassification`, `.from_pretrained()`, `.save_pretrained()`
  - Model outputs: `outputs.loss`, `outputs.logits`
  - `Adafactor` optimizer, `get_linear_schedule_with_warmup`
- [ ] **`datasets`**: `load_dataset('super_glue', name)` in [get_huggingface_superglue.py](../CaPT/superglue/get_huggingface_superglue.py)

## Level 12: Other Libraries

- [ ] **`tqdm`**: progress bars (`tqdm(iterable, total=..., desc=...)`)
- [ ] **`tensorboardX.SummaryWriter`**: `add_scalar()` for logging loss and learning rate
- [ ] **`scikit-learn` metrics**: `f1_score(average='macro')`, `matthews_corrcoef`
- [ ] **`scipy.stats.spearmanr`**

---

## Suggested Learning Path

1. **Weeks 1–2:** Levels 1–5 (core Python). Practise by reading [superglue_loader_decoder.py](../CaPT/src/superglue_loader_decoder.py).
2. **Week 3:** Levels 6–8 (classes, modules, standard library). Read [dataset.py](../CaPT/src/dataset.py) and the `LlamaClassificationDataset` class.
3. **Weeks 4–5:** Levels 9–10 (NumPy + PyTorch). Do the official PyTorch "60-minute blitz" tutorial, then read the `CaPT` class in [modeling_llama.py](../CaPT/transformers/models/llama/modeling_llama.py) (search for `class CaPT`).
4. **Week 6:** Levels 11–12 (Hugging Face). Then read [train_llama.py](../CaPT/capt/train_llama.py) from top to bottom. You should understand every line.

## Self-Check: Can You Explain These Lines?

If you can explain each of these, you are ready for the codebase:

```python
self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
torch.mean(sequence, dim=1, keepdim=True)
hidden_states[:, :self.prompt_len, :] = generated_prompt
feed_to_gpt = hidden_states.detach()
{v: k for k, v in self.label_map.items()}
sum(p.numel() for p in model.parameters() if p.requires_grad)
collate_fn=lambda batch: pad_sequence_collate(batch, tokenizer)
f"{lr=} {batch_size=}"
((premise_joiner, choice1), (_, choice2)) = input_seq
```
