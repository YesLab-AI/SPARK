import os
import json
import webbrowser

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4)

import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
import joblib
import gc
import tempfile
import subprocess
import random
import ctypes
import hashlib
from transformers import AutoModelForCausalLM, T5EncoderModel, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from peft import PeftModel, PeftConfig
from pathlib import Path
from tqdm import tqdm
from torch.amp import autocast
import warnings
from huggingface_hub import login
import argparse
from getpass import getpass

try:
    import transformers.tokenization_utils_base as tub
    _original_setattr = tub.PreTrainedTokenizerBase.__setattr__
    def _safe_setattr(self, key, value):
        try:
            _original_setattr(self, key, value)
        except AttributeError:
            pass
    tub.PreTrainedTokenizerBase.__setattr__ = _safe_setattr

    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein
    from esm.utils.constants.models import ESM3_OPEN_SMALL
    
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
    def _safe_getattr(self, key):
        if key.endswith("_token"):
            val = self.__dict__.get(f"_{key}")
            if val is not None:
                return str(val)
            esm_defaults = {
                "mask_token": "<mask>", 
                "cls_token": "<cls>", 
                "pad_token": "<pad>", 
                "eos_token": "<eos>",
                "bos_token": "<bos>",
                "unk_token": "<unk>"
            }
            if key in esm_defaults:
                return esm_defaults[key]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    EsmSequenceTokenizer.__getattr__ = _safe_getattr

except ImportError:
    pass

warnings.filterwarnings("ignore")

BASE_MODEL_ID = "progen2-large"
SFT_ADAPTER_PATH = Path("models/sft_lora_new")
DATA_PATH = Path("data/data_for_sft.csv")
ORACLE_BASE_ID = "ankh3-large"
ADAPTER_PATH = Path("models/ankh3_adalora_finetune")
ORACLE_DIR = Path("results/final_oracle")
ORACLE_WEIGHTS = ORACLE_DIR / "final_e2e_oracle.pth"
SCALER_Y_PATH = ORACLE_DIR / "scaler_y.pkl"
SIGNALP_EXE = Path(os.environ.get("SIGNALP6_PATH", "signalp6"))
CS_TOLERANCE = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 1536
GEN_BATCH_SIZE = 64
ORACLE_BATCH_SIZE = 64
TARGET_GENERATE = 1500
TARGET_FILTER = 1000

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def release_system_memory():
    torch.cuda.empty_cache()
    gc.collect()
    if os.name == 'nt':
        try:
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
        except Exception:
            pass

class CnnPoolingLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        x = F.relu(self.conv(x))
        if mask is not None: 
            mask = mask.unsqueeze(1)
            x = x.masked_fill(mask == 0, -1e4)
        return torch.max(x, dim=2)[0]

class SwiGLUHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = 1024
        self.w_gate = nn.Linear(input_dim, hidden_dim)
        self.w_transform = nn.Linear(input_dim, hidden_dim)
        self.ln_glu = nn.LayerNorm(hidden_dim)
        self.w_out = nn.Linear(hidden_dim, 256)
        self.ln_out = nn.LayerNorm(256)
        self.predictor = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        gate = self.w_gate(x).float()
        transform = self.w_transform(x).float()
        glu_out = F.silu(gate) * transform
        glu_out = self.ln_glu(glu_out.to(x.dtype)) 
        h = self.dropout(F.gelu(self.ln_out(self.w_out(self.dropout(glu_out)))))
        return self.predictor(h)

class FrozenEncoderOracle(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.pooler = CnnPoolingLayer(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = SwiGLUHead(hidden_dim)
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = encoder_outputs.last_hidden_state
        pooled_features = self.pooler(hidden_states, attention_mask)
        normalized_pooled = self.norm(pooled_features)
        return self.head(normalized_pooled)

def compute_hybrid_oracle_score(sequence: str, neural_network_score: float) -> float:
    seq_clean = re.sub(r'[^A-Z]', '', sequence.upper())
    
    if len(seq_clean) < 10:
        return 0.0
        
    penalty = 0.0
    
    if len(seq_clean) > 2:
        if seq_clean[1] not in ['K', 'R'] or seq_clean[2] not in ['K', 'R']:
            penalty -= 1.0  
            
    if len(seq_clean) > 3 and seq_clean[3] in ['K', 'R']:
        penalty -= 0.3     

    if len(seq_clean) >= 3:
        if seq_clean[-3] != 'A' or seq_clean[-1] != 'A':
            penalty -= 1.0  
            
    final_score = neural_network_score + penalty
    return float(max(0.0, final_score))

class FastAminoAcidLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, min_len=18, max_len=60, max_consecutive=2):
        self.tokenizer = tokenizer
        self.min_len = min_len
        self.max_len = max_len
        self.max_consecutive = max_consecutive
        self.eos_token_id = tokenizer.eos_token_id
        vocab = tokenizer.get_vocab()
        special_ids = set(tokenizer.all_special_ids)
        self.vocab_size = max(vocab.values()) + 1
        self.token_lengths = torch.zeros(self.vocab_size, dtype=torch.long)
        self.always_banned = []
        self.aas = "ACDEFGHIKLMNPQRSTVWY"
        
        char_to_idx = {aa: i for i, aa in enumerate(self.aas)}
        self.token_to_char_idx = torch.full((self.vocab_size,), -1, dtype=torch.long)
        self.ban_dict = {aa: [] for aa in self.aas}
        
        for token, tid in vocab.items():
            if tid in special_ids: continue
            decoded = tokenizer.decode([tid], skip_special_tokens=True).upper()
            clean_str = re.sub(r'[^A-Z]', '', decoded)
            self.token_lengths[tid] = len(clean_str)
            if len(clean_str) == 1:
                char = clean_str
                if char in char_to_idx:
                    self.token_to_char_idx[tid] = char_to_idx[char]
                    self.ban_dict[char].append(tid)
            elif len(clean_str) == 0 and tid != self.eos_token_id:
                self.always_banned.append(tid)
                
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            self.always_banned.append(tokenizer.pad_token_id)
            
        self.char_idx_to_banned = torch.zeros((len(self.aas), self.vocab_size), dtype=torch.bool)
        for char, tids in self.ban_dict.items():
            if char in char_to_idx:
                idx = char_to_idx[char]
                valid_tids = [t for t in tids if t < self.vocab_size]
                self.char_idx_to_banned[idx, valid_tids] = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = scores.device
        vocab_size = scores.shape[-1]
        
        if self.token_lengths.device != device:
            self.token_lengths = self.token_lengths.to(device)
            self.token_to_char_idx = self.token_to_char_idx.to(device)
            self.char_idx_to_banned = self.char_idx_to_banned.to(device)
            
        if self.token_lengths.size(0) != vocab_size:
            new_lengths = torch.full((vocab_size,), 9999, dtype=torch.long, device=device)
            min_len_pad = min(self.token_lengths.size(0), vocab_size)
            new_lengths[:min_len_pad] = self.token_lengths[:min_len_pad]
            self.token_lengths = new_lengths
            new_token_to_char = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
            new_token_to_char[:min_len_pad] = self.token_to_char_idx[:min_len_pad]
            self.token_to_char_idx = new_token_to_char
            new_char_idx_to_banned = torch.zeros((len(self.aas), vocab_size), dtype=torch.bool, device=device)
            new_char_idx_to_banned[:, :min_len_pad] = self.char_idx_to_banned[:, :min_len_pad]
            self.char_idx_to_banned = new_char_idx_to_banned
            
        if self.always_banned:
            valid_banned = [tid for tid in self.always_banned if tid < vocab_size]
            if valid_banned:
                scores[:, valid_banned] = -float('inf')
                
        batch_lengths = self.token_lengths[input_ids].sum(dim=1)
        current_len = input_ids.shape[1]
            
        too_short_mask = batch_lengths < self.min_len
        if too_short_mask.any():
            scores[too_short_mask, self.eos_token_id] = -float('inf')
            
        allowed_new_len = torch.clamp(self.max_len - batch_lengths, min=0)
        too_long_mask = self.token_lengths.unsqueeze(0) > allowed_new_len.unsqueeze(1)
        scores.masked_fill_(too_long_mask, -float('inf'))
        
        done_mask = batch_lengths >= self.max_len
        if done_mask.any():
            scores[done_mask, :] = -float('inf')
            scores[done_mask, self.eos_token_id] = 0.0
            
        if current_len >= 2:
            last_ids = input_ids[:, -1]
            prev_ids = input_ids[:, -2]
            valid_last = last_ids < vocab_size
            valid_last &= prev_ids < vocab_size 
            last_char_idx = torch.where(valid_last, self.token_to_char_idx[last_ids], torch.tensor(-1, device=device))
            prev_char_idx = torch.where(valid_last, self.token_to_char_idx[prev_ids], torch.tensor(-1, device=device))
            match_mask = (last_char_idx == prev_char_idx) & (last_char_idx >= 0)
            if match_mask.any():
                safe_char_idx = torch.where(match_mask, last_char_idx, torch.tensor(0, device=device))
                banned_mask = self.char_idx_to_banned[safe_char_idx] 
                banned_mask = banned_mask & match_mask.unsqueeze(1)
                scores.masked_fill_(banned_mask, -float('inf'))
                
        return scores

def run_signalp6_strict_batch(sps, payload_seq):
    sp_results, sp_types, cs_positions = [0] * len(sps), [""] * len(sps), [-1] * len(sps)
    cmd_exe = str(SIGNALP_EXE) if SIGNALP_EXE.exists() else "signalp6"
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "input.fasta")
        out_dir = os.path.join(tmpdir, "output")
        with open(fasta_path, "w") as f:
            for i, sp in enumerate(sps): f.write(f">seq_{i}\n{sp + payload_seq}\n")
        try:
            subprocess.run([cmd_exe, "--fastafile", fasta_path, "--output_dir", out_dir, "--format", "none", "--organism", "other", "--mode", "fast"], capture_output=True, text=True, check=True)
            res_path = os.path.join(out_dir, "prediction_results.txt")
            if os.path.exists(res_path):
                with open(res_path, "r") as f:
                    for line in f:
                        if line.startswith("#"): continue
                        cols = line.strip().split("\t")
                        idx = int(cols[0].replace("seq_", ""))
                        sp_type_str = cols[1].strip()
                        sp_types[idx] = sp_type_str
                        match = re.search(r"CS pos: (\d+)", line)
                        cp = int(match.group(1)) if match else -1
                        cs_positions[idx] = cp
                        if sp_type_str.upper() != "OTHER" and abs(cp - len(sps[idx])) <= CS_TOLERANCE: 
                            sp_results[idx] = 1
        except Exception: 
            pass
    return sp_results, sp_types, cs_positions

def generate_dashboard(df: pd.DataFrame, html_path: str):
    valid_rate = df['Valid'].mean() * 100 if not df.empty else 0.0
    total_candidates = len(df)
    json_data = df.to_json(orient='records')
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SPARK Interactive Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ background-color: #f4f6f9; font-family: 'Segoe UI', system-ui, sans-serif; }}
        .header-bar {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 20px 0; margin-bottom: 30px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        .card {{ border: none; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 24px; transition: transform 0.2s; }}
        .card:hover {{ transform: translateY(-2px); }}
        .card-header {{ background-color: #fff; border-bottom: 1px solid #f0f0f0; font-weight: 600; font-size: 1.1rem; border-radius: 12px 12px 0 0 !important; padding: 15px 20px; color: #1e3c72; }}
        .table-responsive {{ max-height: 500px; overflow-y: auto; }}
        .table thead th {{ position: sticky; top: 0; background: #f8f9fa; z-index: 1; border-bottom: 2px solid #dee2e6; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.5px; }}
        .badge-success {{ background-color: #10b981; color: white; padding: 6px 12px; border-radius: 20px; font-weight: 500; font-size: 0.85rem; }}
        .badge-danger {{ background-color: #ef4444; color: white; padding: 6px 12px; border-radius: 20px; font-weight: 500; font-size: 0.85rem; }}
        .btn-download {{ background-color: #3b82f6; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; display: inline-flex; align-items: center; gap: 8px; font-size: 1.1rem; }}
        .btn-download:hover {{ background-color: #2563eb; color: white; box-shadow: 0 4px 12px rgba(37,99,235,0.2); transform: translateY(-2px); }}
        .stat-value {{ font-size: 3rem; font-weight: 700; color: #1e293b; line-height: 1.1; }}
        .stat-label {{ color: #64748b; font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }}
        .seq-cell {{ font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; font-size: 1rem; color: #0f172a; }}
    </style>
</head>
<body>

<div class="header-bar">
    <div class="container-fluid px-4">
        <h2 class="m-0 fw-bold">SPARK Output Dashboard</h2>
        <p class="m-0 mt-1 opacity-75">Offline Interactive Analysis Report for Secretory Peptide Candidates</p>
    </div>
</div>

<div class="container-fluid px-4">
    <div class="row">
        <div class="col-md-4">
            <div class="row">
                <div class="col-6">
                    <div class="card text-center h-100">
                        <div class="card-body d-flex flex-column justify-content-center py-4">
                            <div class="stat-value text-primary">{total_candidates}</div>
                            <div class="stat-label mt-2">Candidates Generated</div>
                        </div>
                    </div>
                </div>
                <div class="col-6">
                    <div class="card text-center h-100">
                        <div class="card-body d-flex flex-column justify-content-center py-4">
                            <div class="stat-value text-success">{valid_rate:.2f}%</div>
                            <div class="stat-label mt-2">SignalP Valid Rate</div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="card mt-4" style="background: linear-gradient(to right, #ffffff, #f8fafc);">
                <div class="card-body text-center p-4">
                    <h5 class="mb-3 text-dark fw-bold">Data Export</h5>
                    <button class="btn btn-download w-100 justify-content-center" onclick="downloadCSV()">
                        <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                        Save Data as CSV
                    </button>
                </div>
            </div>
        </div>
        
        <div class="col-md-8">
            <div class="card h-100">
                <div class="card-header">
                    Sequence Length vs. Final Prediction Score
                </div>
                <div class="card-body">
                    <canvas id="scoreChart" style="max-height: 250px;"></canvas>
                </div>
            </div>
        </div>
    </div>

    <div class="card mt-4 mb-5">
        <div class="card-header d-flex justify-content-between align-items-center">
            <span>Ranked Sequence Candidates</span>
            <span class="badge bg-primary rounded-pill py-2 px-3">Sorted by Hybrid Score</span>
        </div>
        <div class="card-body p-0">
            <div class="table-responsive">
                <table class="table table-hover align-middle m-0">
                    <thead>
                        <tr>
                            <th class="ps-4">Rank</th>
                            <th>Sequence</th>
                            <th>Len</th>
                            <th>Valid</th>
                            <th>SignalP Type</th>
                            <th>ESM3 Loss</th>
                            <th class="pe-4">Final Score</th>
                        </tr>
                    </thead>
                    <tbody id="table-body">
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</div>

<script>
    const data = {json_data};
    
    const tbody = document.getElementById('table-body');
    data.forEach((row, index) => {{
        const tr = document.createElement('tr');
        const badgeClass = row.Valid ? 'badge-success' : 'badge-danger';
        
        tr.innerHTML = `
            <td class="ps-4 text-muted fw-bold">#${{index + 1}}</td>
            <td class="seq-cell fw-medium">${{row.SP_Sequence}}</td>
            <td>${{row.Length}}</td>
            <td><span class="${{badgeClass}}">${{row.Valid}}</span></td>
            <td><span class="badge bg-secondary">${{row.SignalP_Type}}</span></td>
            <td class="text-muted">${{parseFloat(row.ESM3_Loss).toFixed(4)}}</td>
            <td class="pe-4 fw-bold" style="color: #0f172a; font-size: 1.1rem;">${{parseFloat(row.Final_Score).toFixed(4)}}</td>
        `;
        tbody.appendChild(tr);
    }});

    function downloadCSV() {{
        const headers = "SP_Sequence,Length,Valid,SignalP_Type,ESM3_Loss,Final_Score";
        const rows = data.map(r => `${{r.SP_Sequence}},${{r.Length}},${{r.Valid}},${{r.SignalP_Type}},${{r.ESM3_Loss}},${{r.Final_Score}}`).join("\\n");
        const csvContent = headers + "\\n" + rows;
        
        const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.setAttribute("href", url);
        link.setAttribute("download", "designed_peptides.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }}

    const ctx = document.getElementById('scoreChart').getContext('2d');
    new Chart(ctx, {{
        type: 'scatter',
        data: {{
            datasets: [{{
                label: 'Generated Candidates',
                data: data.map(d => ({{x: d.Length, y: d.Final_Score}})),
                backgroundColor: 'rgba(59, 130, 246, 0.6)',
                borderColor: '#2563eb',
                borderWidth: 1.5,
                pointRadius: 6,
                pointHoverRadius: 9
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
                x: {{ 
                    title: {{ display: true, text: 'Sequence Length (Amino Acids)', font: {{ weight: 'bold' }} }},
                    grid: {{ color: '#f1f5f9' }}
                }},
                y: {{ 
                    title: {{ display: true, text: 'Final Oracle Score', font: {{ weight: 'bold' }} }},
                    grid: {{ color: '#f1f5f9' }}
                }}
            }},
            plugins: {{ legend: {{ display: false }} }},
            interaction: {{ mode: 'nearest', intersect: true }}
        }}
    }});
</script>
</body>
</html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

def process_design(payload_seq, hf_token):
    dynamic_seed = int(hashlib.md5(payload_seq.encode('utf-8')).hexdigest(), 16) % (2**32)
    set_seed(dynamic_seed)
    
    if hf_token: 
        login(token=hf_token)
    sft_seqs = set()
    if DATA_PATH.exists():
        try:
            df_sft = pd.read_csv(DATA_PATH)
            if 'sequence' in df_sft.columns:
                raw_seqs = df_sft['sequence'].dropna().astype(str).str.upper().tolist()
                sft_seqs = {re.sub(r'[^A-Z]', '', s) for s in raw_seqs}
        except Exception:
            pass
            
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokenizer.padding_side = "left"
    
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True)
    
    if SFT_ADAPTER_PATH.exists():
        base_model = PeftModel.from_pretrained(base_model, str(SFT_ADAPTER_PATH)).merge_and_unload()
        
    gen_model = base_model.to(DEVICE).eval()
    gen_model.config.use_cache = True
    input_ids = tokenizer("M", return_tensors="pt").input_ids.to(DEVICE).repeat(GEN_BATCH_SIZE, 1)
    lp = LogitsProcessorList([FastAminoAcidLogitsProcessor(tokenizer, min_len=18, max_len=60, max_consecutive=2)])
    generated_candidates = []
    generated_set = set()
    pbar = tqdm(total=TARGET_GENERATE, desc="Generating Candidates")
    
    while len(generated_candidates) < TARGET_GENERATE:
        with torch.no_grad(): 
            outputs = gen_model.generate(
                input_ids=input_ids, 
                max_new_tokens=60, 
                do_sample=True, 
                top_p=0.8, 
                top_k=10, 
                temperature=0.9, 
                repetition_penalty=1.2, 
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id, 
                logits_processor=lp,
                use_cache=True 
            )
            decoded = tokenizer.batch_decode(outputs[:, 1:], skip_special_tokens=True)
            for s in decoded:
                clean_s = "M" + re.sub(r'[^A-Z]', '', s.upper())
                if 16 <= len(clean_s) <= 40 and clean_s not in generated_set and clean_s not in sft_seqs:
                    generated_set.add(clean_s)
                    generated_candidates.append(clean_s)
                    pbar.update(1)
                    if len(generated_candidates) >= TARGET_GENERATE:
                        break
    pbar.close()
    
    del gen_model, base_model, tokenizer
    release_system_memory()

    esm3_model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=DEVICE).eval()
    seq_loss_pairs = []
    ESM_BATCH_SIZE = 32
    try:
        esm_pad_id = esm3_model.tokenizer.pad_token_id
    except AttributeError:
        esm_pad_id = 0
        
    for i in tqdm(range(0, len(generated_candidates), ESM_BATCH_SIZE), desc="ESM-3 Batched Scoring"):
        batch_seqs = generated_candidates[i:i+ESM_BATCH_SIZE]
        batch_encoded = [esm3_model.encode(ESMProtein(sequence=seq + payload_seq)).sequence for seq in batch_seqs]
        max_len = max(t.size(0) for t in batch_encoded)
        padded_tensors = []
        for t in batch_encoded:
            if t.size(0) < max_len:
                padding = torch.full((max_len - t.size(0),), esm_pad_id, dtype=t.dtype, device=t.device)
                padded_tensors.append(torch.cat([t, padding]))
            else:
                padded_tensors.append(t)
        batch_tokens = torch.stack(padded_tensors).to(DEVICE)
        with torch.no_grad():
            with autocast('cuda'):
                output = esm3_model(sequence_tokens=batch_tokens)
        for j, original_t in enumerate(batch_encoded):
            seq_len = original_t.size(0)
            item_logits = output.sequence_logits[j, :seq_len, :]
            item_targets = original_t.to(DEVICE)
            loss = F.cross_entropy(item_logits.contiguous().view(-1, item_logits.size(-1)), item_targets.view(-1)).item()
            seq_loss_pairs.append((batch_seqs[j], loss))
            
    del esm3_model
    release_system_memory()

    seq_loss_pairs.sort(key=lambda x: x[1])
    top_candidates = seq_loss_pairs[:TARGET_FILTER]
    candidate_list = [x[0] for x in top_candidates]
    esm_scores = [x[1] for x in top_candidates]

    oracle_tokenizer = AutoTokenizer.from_pretrained(ORACLE_BASE_ID, legacy=False)
    oracle_base = T5EncoderModel.from_pretrained(ORACLE_BASE_ID)
    if ADAPTER_PATH.exists(): 
        peft_config = PeftConfig.from_pretrained(ADAPTER_PATH)
        peft_config.task_type = "FEATURE_EXTRACTION"
        model_with_prior = PeftModel.from_pretrained(oracle_base, str(ADAPTER_PATH), config=peft_config)
        oracle_base = model_with_prior.merge_and_unload()
        
    oracle_model = FrozenEncoderOracle(oracle_base, HIDDEN_DIM).to(DEVICE)

    oracle_model.load_state_dict(torch.load(ORACLE_WEIGHTS, map_location=DEVICE), strict=False)
    oracle_model.eval()
    scaler_y = joblib.load(SCALER_Y_PATH)
    oracle_scores = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(candidate_list), ORACLE_BATCH_SIZE), desc="Oracle Scoring"):
            batch_seqs = candidate_list[i:i+ORACLE_BATCH_SIZE]
            tok_in = oracle_tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            with autocast('cuda'): 
                preds = oracle_model(tok_in['input_ids'], tok_in['attention_mask']).cpu().numpy()
            
            preds_flat = preds.flatten().tolist()
            hybrid_scaled_scores = []
            for seq, p_val in zip(batch_seqs, preds_flat):
                p_val_float = float(p_val)
                hybrid_val = compute_hybrid_oracle_score(seq, p_val_float)
                hybrid_scaled_scores.append(hybrid_val)
            
            hybrid_scaled_arr = np.array(hybrid_scaled_scores).reshape(-1, 1)
            scores_inv = scaler_y.inverse_transform(hybrid_scaled_arr).flatten()
            oracle_scores.extend(scores_inv.tolist())
            
    del oracle_model, oracle_base, oracle_tokenizer
    release_system_memory()

    v_results, v_types, v_pos = [], [], []
    CHUNK_SIZE = 100
    for i in tqdm(range(0, len(candidate_list), CHUNK_SIZE), desc="SignalP Validation (Chunked)"):
        chunk_seqs = candidate_list[i : i + CHUNK_SIZE]
        res, typ, pos = run_signalp6_strict_batch(chunk_seqs, payload_seq)
        v_results.extend(res)
        v_types.extend(typ)
        v_pos.extend(pos)

    results = []
    min_len = min(len(candidate_list), len(v_results))
    for i in range(min_len):
        results.append({
            'SP_Sequence': candidate_list[i], 
            'Length': len(candidate_list[i]), 
            'Valid': bool(v_results[i]),
            'SignalP_Type': v_types[i], 
            'ESM3_Loss': esm_scores[i],
            'Final_Score': oracle_scores[i]
        })
        
    df = pd.DataFrame(results).sort_values(['Final_Score', 'ESM3_Loss'], ascending=[False, True])
    csv_path = "designed_peptides.csv"
    df.to_csv(csv_path, index=False)
    
    html_path = "SPARK_Dashboard.html"
    generate_dashboard(df, html_path)
    
    valid_rate = df['Valid'].mean() * 100 if not df.empty else 0.0
    metrics_report = f"Top {TARGET_FILTER} SignalP Valid Rate: {valid_rate:.2f}%"
    return df.head(10), csv_path, html_path, metrics_report

def validate_payload(payload: str) -> str:
    payload = payload.strip().upper()
    if not payload or not re.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", payload):
        raise ValueError("Invalid sequence")
    return payload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    raw_payload = args.payload if args.payload is not None else input("Payload protein sequence: ")
    try:
        payload = validate_payload(raw_payload)
    except ValueError as e:
        print(f"Error: {e}")
        return

    token = args.token
    if token is None:
        token = input("HuggingFace token: ").strip()

    try:
        top10_df, csv_path, html_path, metrics = process_design(payload, token)
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 60)
    print(metrics)
    print(f"Full results saved to: {csv_path}")
    print(f"Dashboard saved to: {html_path}")
    print("\nTop 10 candidates:")
    print(top10_df.to_string(index=False))
    print("=" * 60)

    try:
        abs_path = os.path.abspath(html_path)
        dashboard_url = f"file://{abs_path}"
        webbrowser.open(dashboard_url)
    except Exception:
        pass

if __name__ == "__main__":
    main()