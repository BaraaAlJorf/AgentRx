
# MedPrompt Experiment (kNN Few-Shot + CoT + Pass@K)

This README documents how to run the **MedPrompt** setup implemented in this repo:
1) **Preprocessing phase**: build a memory of *correctly solved* training exemplars (with stored model reasoning + answer) and their text embeddings.
2) **Inference phase**: for each test patient, embed its query text, retrieve **k nearest neighbors** from the memory using **cosine similarity** on the query text, then run CoT and aggregate **Pass@K** votes by re-running the same HF model multiple times.
	- In the original MedPrompt paper, the models in ensembling differed only through choice shuffling. Since choice shuffling is not applicable to our task, ensembling here is essentially a Pass@K with the same model. However, diversity through regenerated chains of thought is present.
---

## Files involved

- `AgentiCDS/agent_architectures.py`
	- MedPrompt inference entrypoint: `initialize_agent_setup(...)-> _run_medprompt_batch(...)`
	- Retrieval helper: `_get_medprompt_exemplars(...)` (loads memory once + retrieves kNN)
	- Query text builder for embeddings: `_medprompt_query_text(...)`
	- Prompt construction: `_build_medprompt_prompt(...)`

- `AgentiCDS/medprompt_memory.py`
	- `HFTextEmbedder`: HF encoder + mean pooling + L2 normalization
	- `MedPromptMemory`: stores `examples.jsonl` + `embeddings.npy`, retrieves kNN by cosine similarity

- `AgentiCDS/medprompt_preprocessing.py`
	- Builds the memory directory (`examples.jsonl`, `embeddings.npy`, `meta.json`)
---

## Preprocessing phase: build MedPrompt memory

### What it does
For each training patient:
1) Run **CoT reasoning** (`prompt_type="cot_reasoning"`)
2) Run **final answer** conditioned on reasoning (`prompt_type="cot_answer"`)
3) If predicted label matches ground truth:
	 - embed the query text (`_medprompt_query_text`)
	 - store exemplar fields + `model_reasoning` + `model_answer`
	 - save embedding vector into `embeddings.npy`

### Command
From the repo root:

```bash
python medprompt_preprocessing \
	--data_path /path/to/train.jsonl \
	--model_id YOUR_HF_MODEL_ID \
	--medprompt_memory_dir /path/to/output/medprompt_memory \
	--batch_size 4 \
	--num_workers 0 \
	--modalities ehr-cxr-rr-ps \
	--medprompt_embed_model_id BAAI/bge-small-en-v1.5 \
	--medprompt_reasoning_tokens 256 \
	--medprompt_do_sample \
	--medprompt_temperature 0.7 \
	--medprompt_top_p 0.9
```

#### Note
- When performaning a new experiment with different choice of allowed modalities, the preprocessing phase needs to be re-run, i.e. the memory cache needs to be re-created.

### Output files
In `--medprompt_memory_dir`:
- `examples.jsonl` : one JSON exemplar per line (includes stored reasoning + answer)
- `embeddings.npy` : float32 array of shape `[N, D]` (L2-normalized)
- `meta.json` : metadata including embedding model id and dimension

---

## Inference phase: run MedPrompt agent

### What it does per batch
For each test patient:
1) Embed query text and retrieve **k = `--num_shots`** nearest exemplars (i.e. reasoning + answer) via cosine similarity
2) Build a prompt containing:
	 - retrieved exemplars (reconstructed question + retrieved reasoning and answer)
	 - the target patient
3) Run **Pass@K**:
	 - for the current patient, repeat reasoning+answer **K = `--medprompt_ensemble_size`** times
	 - aggregate by majority vote (`final_prob = yes_votes / K`)

### Key knobs
- `--num_shots` : number of retrieved neighbors (k in kNN)
- `--medprompt_ensemble_size` : Pass@K size (number of repeated runs)
- `--medprompt_memory_dir ...` (directory created in preprocessing)
- `--medprompt_embed_model_id ...` (must match what you used, or at least produce same embedding dim)
---

## Notes / common pitfalls

### 1) Match generation settings between preprocessing and inference (IMPORTANT)
MedPrompt stores exemplars that were generated using specific decoding settings during **preprocessing** (see `AgentiCDS/medprompt_preprocessing.py`). For consistency with the paper and best results, the **inference-time generation settings should match** what you used when building the memory, especially:

- `--medprompt_reasoning_tokens`
- `--medprompt_do_sample`
- `--medprompt_temperature`
- `--medprompt_top_p`

If you change any of these, re-run preprocessing to rebuild the memory (recommended) or expect retrieval + prompting behavior to be less consistent.

**MedPrompt paper uses do_sample=True by default, so for consistency, use True for this argument. Also, in the script, the values for reasoning_tokens, do_sample, temperature, top_p are defaulted to 256, True, 0.7, 0.9 respectively, so you can skip these args if you want those defaults.**

### 2) `cxr_image_path` must be present in exemplars
Exemplars saved to JSONL do not include `pil_image` (not serializable). They rely on `cxr_image_path` so `_build_prompt_content` can `Image.open(...)` at inference.

### 3) Memory not found
If inference throws:
- `FileNotFoundError: MedPrompt memory not found in: ...`
ensure `--medprompt_memory_dir` contains `examples.jsonl` and `embeddings.npy`.