import traceback, os
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

model_id = "microsoft/llava-med-v1.5-mistral-7b"
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

load_kwargs = {
    "trust_remote_code": True,
    "torch_dtype": "auto",
    "device_map": "auto",
    "use_auth_token": hf_token
}

try:
    print("-> Loading processor...")
    proc = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_auth_token=hf_token
    )
    print("-> Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    print("OK: model class =", model.__class__.__name__)
except Exception:
    traceback.print_exc()

