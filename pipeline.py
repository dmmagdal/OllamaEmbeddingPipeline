# pipeline.py


from dataclasses import dataclass, field
import json
import os
import pathlib
import shutil
import subprocess
from typing import Dict

import httpx
from transformers import AutoModel, AutoTokenizer


@dataclass
class ModelConfig:
	model_id: str          # HuggingFace model ID
	name: str              # short name for Ollama
	storage_dir: str
	dims: int
	max_tokens: int
	quants: list[str] = field(default_factory=lambda: ["f32", "f16", "q8_0", "q4_k_m"])

# Open the config.json file to identify all models for this pipeline.
with open("config.json", "r") as f:
	models = json.load(f)["models"]

# Get the model keys and use that to isolate the model configurations.
model_keys = sorted(list(models.keys()))
model_configs = [
	models[key] for key in model_keys
	if "hkunlp" not in key # Exclude hkunlp models because they're encoder-decoder models that require additional attention.
]

# Load the model configurations to the data class.
MODELS = [
	ModelConfig(
		model_id=model_config["model_id"],
		name=model_name,
		storage_dir=model_config["storage_dir"],
		dims=model_config["dims"],
		max_tokens=model_config["max_tokens"],
	) for model_name, model_config in zip(model_keys, model_configs)
]

# Important paths.
MODELS_DIR = pathlib.Path("./models")
LLAMA_CPP  = pathlib.Path("./llama.cpp")

# ── Stage 0: Download models from HF ─────────────────────────
def download_model(model_configs: Dict) -> None:
	for model_config in model_configs:
		# model_config = model_configs[key]
		model_id = model_config["model_id"]
		storage_dir = model_config["storage_dir"]
		cache_dir = model_config["cache_dir"]

		if os.path.exists(storage_dir) and len(os.listdir(storage_dir)) != 0:
			continue

		tokenizer = AutoTokenizer.from_pretrained(
			model_id, cache_dir=cache_dir
		)
		model = AutoModel.from_pretrained(
			model_id, cache_dir=cache_dir, 
		)

		# Save the tokenizer and model to the save path.
		tokenizer.save_pretrained(storage_dir)
		model.save_pretrained(storage_dir)

		# Delete the cache.
		shutil.rmtree(cache_dir)

# ── Stage 1: Convert HF → F32 GGUF (once per model) ──────────
def convert_to_f32(cfg: ModelConfig) -> pathlib.Path:
	out_dir = MODELS_DIR / cfg.name
	out_dir.mkdir(parents=True, exist_ok=True)
	f32_path = out_dir / "model_f32.gguf"

	if f32_path.exists():
		print(f"  ↩ F32 already exists, skipping conversion for {cfg.name}")
		return f32_path

	subprocess.run([
		"python", str(LLAMA_CPP / "convert_hf_to_gguf.py"),
		# cfg.model_id,
		cfg.storage_dir,
		"--outfile", str(f32_path),
		"--outtype", "f32"
	], check=True)

	assert f32_path.exists() and f32_path.stat().st_size > 500_000, \
		f"F32 GGUF missing or too small for {cfg.name}"
	print(f"  ✓ Converted {cfg.name} → F32")
	return f32_path

# ── Stage 2: Quantize from F32 → each target format ──────────
def quantize(cfg: ModelConfig, f32_path: pathlib.Path) -> dict[str, pathlib.Path]:
	out_dir = f32_path.parent
	quant_paths = {"f32": f32_path}

	for quant in cfg.quants:
		if quant == "f32":
			continue  # already have it

		out_path = out_dir / f"model_{quant}.gguf"

		if out_path.exists():
			print(f"  ↩ {quant} already exists, skipping")
			quant_paths[quant] = out_path
			continue

		if quant == "f16":
			# convert_hf_to_gguf can do f16 directly — faster than llama-quantize
			subprocess.run([
				"python", str(LLAMA_CPP / "convert_hf_to_gguf.py"),
				# cfg.model_id,
				cfg.storage_dir,
				"--outfile", str(out_path),
				"--outtype", "f16"
			], check=True)
		else:
			# All integer quants go through llama-quantize
			subprocess.run([
				str(LLAMA_CPP / "build/bin/llama-quantize"),
				str(f32_path),
				str(out_path),
				quant.upper().replace("-", "_")   # q4_k_m → Q4_K_M
			], check=True)

		assert out_path.exists() and out_path.stat().st_size > 100_000, \
			f"Quantized file missing or too small: {out_path}"
		print(f"  ✓ Quantized {cfg.name} → {quant}")
		quant_paths[quant] = out_path

	return quant_paths

# ── Stage 3: Register each quant with Ollama ─────────────────
def register_all(cfg: ModelConfig, quant_paths: dict[str, pathlib.Path]):
	for quant, gguf_path in quant_paths.items():
		ollama_name = f"{cfg.name}-{quant}"   # eg. bert-base-q4_k_m
		modelfile = f"FROM {gguf_path.resolve()}\n"
		mf_path = gguf_path.with_suffix(".Modelfile")
		mf_path.write_text(modelfile)

		subprocess.run(
			["ollama", "create", ollama_name, "-f", str(mf_path)],
			check=True
		)
		print(f"  ✓ Registered → {ollama_name}")

# ── Stage 4: Verify each registered model ────────────────────
def verify_all(cfg: ModelConfig, quant_paths: dict[str, pathlib.Path]):
	listed = subprocess.run(
		["ollama", "list"], capture_output=True, text=True, check=True
	).stdout

	baseline_vec = None  # F32 vector — compare all quants against this

	for quant in quant_paths:
		ollama_name = f"{cfg.name}-{quant}"
		assert ollama_name in listed, f"✗ {ollama_name} not found in ollama list"

		resp = httpx.post(
			"http://localhost:11434/api/embeddings",
			json={"model": ollama_name, "prompt": "the quick brown fox"}
		)
		resp.raise_for_status()
		vec = resp.json()["embedding"]

		# Dim check
		assert len(vec) == cfg.dims, \
			f"✗ {ollama_name}: expected dim {cfg.dims}, got {len(vec)}"

		# Non-zero check
		assert any(v != 0 for v in vec), f"✗ {ollama_name}: zero vector returned"

		# Cosine similarity vs F32 baseline
		if quant == "f32":
			baseline_vec = vec
		elif baseline_vec:
			similarity = cosine_sim(baseline_vec, vec)
			assert similarity > 0.99, \
				f"✗ {ollama_name}: cosine sim vs F32 = {similarity:.4f} (too low)"
			print(f"  ✓ {ollama_name} — dim={len(vec)}, cosine_vs_f32={similarity:.4f}")
			continue

		print(f"  ✓ {ollama_name} — dim={len(vec)}")

def cosine_sim(a: list[float], b: list[float]) -> float:
	dot   = sum(x * y for x, y in zip(a, b))
	mag_a = sum(x ** 2 for x in a) ** 0.5
	mag_b = sum(x ** 2 for x in b) ** 0.5
	return dot / (mag_a * mag_b)

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
	download_model(model_configs)

	for cfg in MODELS:
		print(f"\n{'='*50}\nProcessing: {cfg.name}\n{'='*50}")
		f32_path    = convert_to_f32(cfg)
		quant_paths = quantize(cfg, f32_path)
		register_all(cfg, quant_paths)
		verify_all(cfg, quant_paths)
# ```

# ---

## What This Gets You

# For each model in your config you end up with a clean directory tree and an Ollama name per quant:
# ```
# models/
# └── bert-base/
#     ├── model_f32.gguf
#     ├── model_f16.gguf
#     ├── model_q8_0.gguf
#     ├── model_q4_k_m.gguf
#     ├── model_f32.Modelfile
#     └── ...

# ollama list:
#   bert-base-f32
#   bert-base-f16
#   bert-base-q8_0
#   bert-base-q4_k_m