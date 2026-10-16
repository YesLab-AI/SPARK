# SPARK:Signal Peptide Design Accelerated by Generative and Ranking Models

**Deep learning-guided design of optimal signal peptides for enhanced protein secretion**. It pairs a ProGen2 *generator* with an Ankh3-based *oracle* (scorer),
and adds signal-peptide priors (SignalP 6 + motif rules) and ESM-3 structure
prediction.

## Overview

Four sequential stages — **run them in order**, since each consumes the previous
stage's output.

| Stage | Script | Role | Method |
|------:|--------|------|--------|
| 1 | `01_train_ankh3.py` | Representation prior | AdaLoRA + supervised contrastive fine-tuning of Ankh3 |
| 2 | `02_regression.py` | Oracle (scorer) | Frozen encoder + CNN-pool / SwiGLU regression head (hybrid signal-peptide score) |
| 3 | `03_sft_proGen2.py` | Generator | LoRA SFT of ProGen2-large |
| 4 | `04_generation.py` | Design + evaluate | Generate, score, run SignalP 6 & ESM-3, export candidates (terminal / CLI) |

## Repository structure

```
.
├── 01_train_ankh3.py
├── 02_regression.py
├── 03_sft_proGen2.py
├── 04_generation.py
├── requirements.txt
├── data/                   
├── models/                  # trained adapters       
│   ├── ankh3_adalora_finetune/
│   └── sft_lora_new/
├── results/final_oracle/    # trained adapters 
├── ankh3-large/             # base weights           (obtain separately)
├── progen2-large/           # base weights           (obtain separately)
└── signalp-6.0/             # SignalP 6              (license required)
```

## Installation

**Environment** (Python 3.10, tested on 3.10.19; CUDA GPU required):

```bash
conda create -n gen4sec python=3.10 -y
conda activate gen4sec
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**External assets** (large / licensed — obtain separately):

- `ankh3-large/`, `progen2-large/`, `esmfold_v1/` — base model weights, placed at the repo root.
- **ESM-3** (`EvolutionaryScale/esm3-sm-open-v1`) — gated model. **Each user must supply their own Hugging Face token** (see [Stage 4](#stage-4--hugging-face-token-required)): request access to the model on the Hub, then create a token at <https://huggingface.co/settings/tokens>.
- **SignalP 6.0** — academic license from DTU Health Tech; put `signalp6` on your `PATH`, or set the `SIGNALP6_PATH` environment variable to the executable.

## Data

Place CSVs under `data/`:

| File | Used by | Columns |
|------|---------|---------|
| `classification.csv` | Stage 1 | `sequence`, `labels` (0/1) |
| `regression_data.csv` | Stage 2 | `sequence`, `normalized_value` |
| `data_for_sft.csv` | Stage 3–4 | `sequence` |

## Usage

Run from the repository root:

```bash
python 01_train_ankh3.py   # -> models/ankh3_adalora_finetune/
python 02_regression.py    # -> results/final_oracle/ 
python 03_sft_proGen2.py   # -> models/sft_lora_new/
python 04_generation.py    # terminal pipeline -> designed_peptides.csv
```

Stage 4 requires the outputs of Stages 1–3 (both adapters + the oracle head + scaler).

### Stage 4 — Hugging Face token (required)

Stage 4 uses the gated **ESM-3** model, so **every user must supply their own
Hugging Face token** — there is no shared or default token in the code. One-time setup:

1. Sign in to Hugging Face and request access to `EvolutionaryScale/esm3-sm-open-v1`.
2. Create an access token at <https://huggingface.co/settings/tokens>.

`04_generation.py` runs entirely in the terminal. Provide the token in any one of these ways:

```bash
# Interactive: prompts for the payload sequence and token (token input is hidden)
python 04_generation.py

# Or pass them as command-line arguments:
python 04_generation.py --payload EVQLVESGGG... --token hf_xxx

# Or log in once, then leave the token blank when prompted:
huggingface-cli login
python 04_generation.py
```

Progress is printed in the terminal; the ranked candidates are written to
`designed_peptides.csv`, and the top 10 are shown on screen.

## Pretrained weights

The oracle head and scaler (`results/final_oracle/`) ship with this repo. To run
Stage 4 without retraining, you only need the two adapters under `models/` —
download and unpack them at the repository root:

```bash
pip install gdown
gdown <1CxdJCgBeK4tQqBXvtfFSrxj_iuMZ8Rjj> -O weights.zip
# Windows:      Expand-Archive weights.zip -DestinationPath .
# Linux/macOS:  unzip weights.zip
```

## License

Released under the MIT License. Base models (Ankh3, ProGen2, ESM-3) and SignalP 6
carry their own licenses.

## Acknowledgements

Built on Ankh3, ProGen2, [ESM-3](https://github.com/evolutionaryscale/esm), and
SignalP 6.0, with Hugging Face `transformers`, `peft`, and `trl`.
