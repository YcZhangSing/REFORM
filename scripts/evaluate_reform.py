#!/usr/bin/env python3
"""Evaluate REFORM on ROM metadata files."""

import argparse
import datetime as dt
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Subset
from torchvision.ops.boxes import box_area
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reform.multilabel_metrics import AveragePrecisionMeter
from reform.rom_dataset import OPTIONS, ROMDatasetForEvaluation, load_rom_json


OPTION_LABELS = [
    [0, 0, 0, 0, 0],
    [1, -0.25, -0.25, -0.25, -0.25],
    [-0.25, 1, -0.25, -0.25, -0.25],
    [-0.25, -0.25, 1, -0.25, -0.25],
    [-0.25, -0.25, -0.25, 1, -0.25],
    [1, -0.66, -0.66, -0.66, 1],
    [-0.66, 1, -0.66, -0.66, 1],
    [-0.66, -0.66, 1, -0.66, 1],
    [-0.66, -0.66, -0.66, 1, 1],
    [-0.25, -0.25, -0.25, -0.25, 1],
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate REFORM checkpoints.")
    parser.add_argument("--model-id", required=True, help="Checkpoint or Hugging Face model path.")
    parser.add_argument("--data-files", nargs="+", required=True, help="One or more ROM meta.json files.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--mode", choices=["fast", "explainable"], default="explainable")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-ratio", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def parse_coordinates(text):
    match = re.search(r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>", text)
    if match:
        return torch.tensor([[int(match.group(i)) for i in range(1, 5)]])
    return torch.tensor([[0, 0, 0, 0]])


def box_iou(boxes1, boxes2, test=False):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union
    if test:
        zero_lines = boxes2 == torch.zeros_like(boxes2)
        zero_lines_idx = torch.where(zero_lines[:, 0] == True)[0]
        for idx in zero_lines_idx:
            if all(boxes1[idx, :] < 1e-4):
                iou[idx] = 1
    return iou, union


def get_multi_label(answers, device):
    multi_label = torch.zeros([len(answers), 5], dtype=torch.long).to(device)
    real_label_pos = [i for i, ans in enumerate(answers) if ans.startswith("A.")]
    label_rows = [
        ("A.", [0, 0, 0, 0, 0]),
        ("B.", [1, 0, 0, 0, 0]),
        ("C.", [0, 1, 0, 0, 0]),
        ("D.", [0, 0, 1, 0, 0]),
        ("E.", [0, 0, 0, 1, 0]),
        ("F.", [1, 0, 0, 0, 1]),
        ("G.", [0, 1, 0, 0, 1]),
        ("H.", [0, 0, 1, 0, 1]),
        ("I.", [0, 0, 0, 1, 1]),
        ("J.", [0, 0, 0, 0, 1]),
    ]
    for prefix, row in label_rows:
        pos = [i for i, ans in enumerate(answers) if ans.startswith(prefix)]
        if pos:
            multi_label[pos, :] = torch.tensor(row).to(device)
    return multi_label, real_label_pos


def main():
    args = parse_args()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or Path(args.model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"domain_test_out_{timestamp}.txt"

    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    vectorizer = TfidfVectorizer().fit(OPTIONS)
    option_vectors = vectorizer.transform(OPTIONS).toarray()
    option_labels = [torch.tensor(row).to(device) for row in OPTION_LABELS]

    def log_print(*items):
        print(*items)
        with summary_path.open("a", encoding="utf-8") as f:
            print(*items, file=f)

    def collate_fn(batch):
        images, questions, answers, _fake_image_box, image_paths, captions = zip(*batch)
        inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
        return inputs, answers, image_paths, captions

    def run_batch(inputs):
        generate_kwargs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "return_decoder2_outputs": args.mode == "explainable",
        }
        generated = model.generate(**generate_kwargs)
        if isinstance(generated, dict):
            answer_ids = generated.get("answer", generated.get("sequences"))
            reason_ids = generated.get("reason")
        else:
            answer_ids = generated
            reason_ids = None
        answers = processor.batch_decode(answer_ids, skip_special_tokens=False)
        reasons = (
            processor.batch_decode(reason_ids, skip_special_tokens=False)
            if reason_ids is not None
            else [""] * len(answers)
        )
        return answers, reasons

    def get_best_option(generated_texts):
        generated_vectors = vectorizer.transform(generated_texts).toarray()
        similarities = cosine_similarity(generated_vectors, option_vectors)
        best_option_indices = similarities.argmax(axis=1)
        best_options = [OPTIONS[i] for i in best_option_indices]
        best_similarities = [similarities[i, best_option_indices[i]] for i in range(len(generated_texts))]
        best_multi_labels = torch.stack([option_labels[i] for i in best_option_indices], dim=0)
        pred_label = torch.ones(len(generated_texts), dtype=torch.long).to(device)
        real_label_pos = [i for i, ans in enumerate(best_options) if ans.startswith("A.")]
        pred_label[real_label_pos] = 0
        return best_options, best_similarities, best_multi_labels, pred_label

    def evaluate_loader(test_loader, output_log):
        iou_pred = []
        cls_nums_all = 0
        cls_acc_all = 0
        multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
        multi_label_meter.reset()

        with output_log.open("a", encoding="utf-8") as log_file:
            for inputs, batch_answers, image_paths, captions in tqdm(test_loader, desc="Evaluating"):
                generated_texts, reasons = run_batch(inputs)
                for i, image_path in enumerate(image_paths):
                    log_entry = {
                        "image": image_path,
                        "label_answer": batch_answers[i],
                        "caption": captions[i],
                        "REFORM_reason": reasons[i] if args.mode == "explainable" else "",
                        "REFORM_answer": generated_texts[i],
                    }
                    log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                task_answers = []
                output_coords = torch.zeros((len(generated_texts), 4)).to(device)
                true_coords = torch.zeros((len(generated_texts), 4)).to(device)
                for i, (generated_text, label_answer) in enumerate(zip(generated_texts, batch_answers)):
                    full_answer = re.sub(r"<pad>|<s>|</s>", "", generated_text)
                    if "<loc_" in full_answer:
                        task_answers.append(full_answer.split("Fake area")[0])
                        output_coords[i] = parse_coordinates(full_answer).to(device)
                    else:
                        task_answers.append(full_answer)
                    true_coords[i] = parse_coordinates(label_answer).to(device)

                real_multi_label, real_label_pos = get_multi_label(batch_answers, device)
                real_label = torch.ones(len(generated_texts), dtype=torch.long).to(device)
                real_label[real_label_pos] = 0
                _best_options, _similarities, best_multi_labels, pred_label = get_best_option(task_answers)
                iou, _ = box_iou(output_coords, true_coords.to(device), test=True)

                for iou_value in iou.cpu().tolist():
                    if isinstance(iou_value, (int, float)) and not math.isnan(iou_value) and not math.isinf(iou_value):
                        iou_pred.append(iou_value)
                    else:
                        iou_pred.append(0.0)

                cls_nums_all += len(generated_texts)
                cls_acc_all += torch.sum(real_label == pred_label).item()
                multi_label_meter.add(best_multi_labels, real_multi_label)

        iou_score = sum(iou_pred) / len(iou_pred) if iou_pred else 0.0
        acc_cls = cls_acc_all / cls_nums_all if cls_nums_all else 0.0
        map_score = multi_label_meter.value().mean()
        op, ore, of1, cp, cr, cf1 = multi_label_meter.overall()
        return acc_cls, cls_acc_all, cls_nums_all, op, ore, of1, cp, cr, cf1, iou_score, map_score

    log_print(f"model_id={args.model_id} mode={args.mode} test_ratio={args.test_ratio}")
    for data_file in args.data_files:
        val_data = load_rom_json(data_file, args.dataset_root)
        dataset = ROMDatasetForEvaluation(split="validation", data=val_data)
        if args.test_ratio < 1:
            subset_size = max(1, int(args.test_ratio * len(dataset)))
            dataset = Subset(dataset, random.sample(range(len(dataset)), subset_size))

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            prefetch_factor=None,
        )
        log_path = output_dir / f"reason_predict_log_{Path(data_file).stem}_{timestamp}.jsonl"
        metrics = evaluate_loader(loader, log_path)
        acc_cls, cls_acc_all, cls_nums_all, op, ore, of1, cp, cr, cf1, iou_score, map_score = metrics

        log_print("#######<--record-->###########")
        log_print(f"dataset={data_file}")
        log_print(f"ACC_cls: {acc_cls * 100:.4f} (cls_acc_all: {cls_acc_all}, cls_nums_all: {cls_nums_all})")
        log_print(f"MAP: {float(map_score) * 100:.4f}")
        log_print(f"IoUscore: {iou_score * 100:.4f}")
        log_print(f"OP={op} OR={ore} OF1={of1} CP={cp} CR={cr} CF1={cf1}")
        log_print("END############################################################################################")

    print(f"Summary log saved to {summary_path}")


if __name__ == "__main__":
    main()

