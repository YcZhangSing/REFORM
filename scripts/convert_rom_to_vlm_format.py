#!/usr/bin/env python3
"""Convert ROM metadata into the JSON format consumed by stage-3 GRPO."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reform.rom_dataset import LABEL_TO_OPTION, load_rom_json


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ROM meta.json to VLM/GRPO format.")
    parser.add_argument("--input", required=True, type=Path, help="Input ROM meta.json.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--relative-image-paths",
        action="store_true",
        help="Keep image_path relative to dataset root instead of writing absolute paths.",
    )
    return parser.parse_args()


def build_solution(item):
    answer = LABEL_TO_OPTION[item["fake_cls"]]
    bbox = item.get("fake_image_box") or []
    if "face" in item["fake_cls"] and bbox:
        x1, y1, x2, y2 = bbox
        answer += f"Fake area<loc_{int(x1)}><loc_{int(y1)}><loc_{int(x2)}><loc_{int(y2)}>"
    return answer


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve() if args.dataset_root else None
    data = load_rom_json(args.input, dataset_root)

    converted = []
    for item in data:
        image_path = Path(item["image"])
        if args.relative_image_paths:
            if dataset_root is None:
                raise ValueError("--relative-image-paths requires --dataset-root")
            image_path_value = image_path.relative_to(dataset_root).as_posix()
        else:
            image_path_value = str(image_path)

        converted_item = {
            "id": item["id"],
            "image_path": image_path_value,
            "text": item["text"],
            "fake_cls": item["fake_cls"],
            "fake_image_box": item.get("fake_image_box", []),
            "bbox": item.get("fake_image_box", []),
            "solution": build_solution(item),
        }
        if "Internvl_out_think" in item:
            converted_item["Internvl_out_think"] = item["Internvl_out_think"]
        converted.append(converted_item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Converted {len(converted)} samples -> {args.output}")


if __name__ == "__main__":
    main()
