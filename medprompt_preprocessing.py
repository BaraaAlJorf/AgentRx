import argparse
import numpy as np

from datasets.data_utils import get_data_loader
from agent_architectures import (
    get_model_and_processor,
    _parse_allowed_modalities,
    _build_prompt_content,
    generate_response,
    generate_yes_no_probability,
    _medprompt_query_text,
)
from medprompt_memory import HFTextEmbedder, MedPromptMemory


def build_memory(loader, args):
    model, processor = get_model_and_processor(args)
    allowed_mods = _parse_allowed_modalities(args)

    embedder = HFTextEmbedder(getattr(args, "medprompt_embed_model_id", "BAAI/bge-small-en-v1.5"))
    memory = MedPromptMemory(embedder=embedder)

    max_patients = int(getattr(args, "medprompt_max_patients", -1))
    stored = 0
    seen = 0

    for batch in loader:
        # 1) CoT reasoning prompts
        reasoning_prompts = []
        for patient in batch:
            content, _ = _build_prompt_content(
                patient,
                patient.get("ehr_text", "EHR Data Not Available"),
                allowed_mods,
                prompt_type="cot_reasoning",
            )
            reasoning_prompts.append([{"role": "user", "content": content}])

        reasoning = generate_response(
            reasoning_prompts,
            model,
            processor,
            max_tokens=int(getattr(args, "medprompt_reasoning_tokens", 256)),
            do_sample=bool(getattr(args, "medprompt_do_sample", True)),
            temperature=float(getattr(args, "medprompt_temperature", 0.7)),
            top_p=float(getattr(args, "medprompt_top_p", 0.9)),
        )

        # 2) Final answer prompts conditioned on reasoning
        answer_prompts = []
        for i, patient in enumerate(batch):
            content, _ = _build_prompt_content(
                patient,
                patient.get("ehr_text", "EHR Data Not Available"),
                allowed_mods,
                prompt_type="cot_answer",
                previous_reasoning=reasoning[i],
            )
            answer_prompts.append([{"role": "user", "content": content}])

        texts, probs_yes = generate_yes_no_probability(answer_prompts, model, processor, max_tokens=1)
        pred = (probs_yes > 0.5).astype(np.int32)

        # 3) Store only correct
        for i, patient in enumerate(batch):
            y = int(patient["labels"]["in_hospital_mortality_48hr"])
            if int(pred[i]) != y:
                continue

            # Build the same embedding "question" text used at inference
            qtext = _medprompt_query_text(patient, allowed_mods)
            emb = embedder.embed([qtext])  # [1, D]

            example = {
                "subject_id": patient.get("subject_id"),
                "stay_id": patient.get("stay_id"),
                "patient_summary_text": patient.get("patient_summary_text", ""),
                "radiology_report_text": patient.get("radiology_report_text", ""),
                "ehr_text": patient.get("ehr_text", ""),
                "cxr_image_path": patient.get("cxr_image_path", "CXR not available"),
                "labels": patient.get("labels", {}),
                # MedPrompt additions
                "model_reasoning": reasoning[i],
                "model_answer": texts[i],
            }

            memory.add(example, emb)
            stored += 1

            if max_patients > 0 and stored >= max_patients:
                memory.save(args.medprompt_memory_dir)
                return

        seen += len(batch)

    memory.save(args.medprompt_memory_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--model_id", required=True)

    p.add_argument("--medprompt_memory_dir", required=True)
    p.add_argument("--medprompt_embed_model_id", default="BAAI/bge-small-en-v1.5")

    p.add_argument("--medprompt_max_patients", type=int, default=-1)
    p.add_argument("--medprompt_reasoning_tokens", type=int, default=256)
    p.add_argument("--medprompt_do_sample", action="store_true", default=True) # medprompt uses sampling so use sampling by default
    p.add_argument("--medprompt_temperature", type=float, default=0.7)
    p.add_argument("--medprompt_top_p", type=float, default=0.9)

    # reuse your existing modalities format (e.g. "ehr-cxr-rr-ps")
    p.add_argument("--modalities", default="ehr-cxr-rr-ps")

    args = p.parse_args()

    loader = get_data_loader(args.data_path, args.batch_size, args.num_workers)
    build_memory(loader, args)

if __name__ == "__main__":
    main()