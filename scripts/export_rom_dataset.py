#!/usr/bin/env python3
"""Export the local ROM data into the public repository layout."""

import argparse
import json
import shutil
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/REFORM_ROMdataset"
)

DATASET_SPECS = [
    {
        "split": "train",
        "domain": "NYT",
        "paths": [
            "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/InternVL3_5/NYT_res/nyt_train_think.json",
        ],
    },
    {
        "split": "train",
        "domain": "Guardian",
        "paths": [
            "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/InternVL3_5/Guard_res/guard_train_think.json",
        ],
    },
    {
        "split": "val",
        "domain": "NYT",
        "paths": [
            "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/MDSMv2/NYT/meta/nyt_val.json",
        ],
    },
    {
        "split": "val",
        "domain": "Guardian",
        "paths": [
            "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/MDSMv2/guard/meta/guardian_val.json",
        ],
    },
    {
        "split": "test",
        "domain": "NYT",
        "paths": [
            "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/MDSMv2/NYT/meta/nyt_test.json",
        ],
    },
    {
        "split": "test",
        "domain": "Guardian",
        "paths": [
            "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/MDSMv2/guard/meta/guardian_test.json",
        ],
    },
    {
        "split": "test",
        "domain": "BBC",
        "paths": [
            "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/MDSMv2/bbc/meta/bbc_test.json",
            "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/MDSMv2/bbc/meta/bbc_train.json",
            "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/MDSMv2/bbc/meta/bbc_val.json",
        ],
    },
    {
        "split": "test",
        "domain": "USAToday",
        "paths": [
            "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/MDSMv2/usa/meta/usa_today_test.json",
            "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/MDSMv2/usa/meta/usa_today_train.json",
            "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/MDSMv2/usa/meta/usa_today_val.json",
        ],
    },
    {
        "split": "test",
        "domain": "Wash",
        "paths": [
            "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/MDSMv2/wash/meta/washington_post_test.json",
            "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/MDSMv2/wash/meta/washington_post_train.json",
            "/mnt/da36552c-a636-46f9-9a37-676e692003a2/yuchen/zycDGM4/MDSMv2/wash/meta/washington_post_val.json",
        ],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Export ROM images and metadata.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=int, default=5000)
    return parser.parse_args()


def copy_image(src_path, dst_path, overwrite=False):
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if not src_path.is_file():
        raise FileNotFoundError(src_path)
    if dst_path.exists() and not overwrite and dst_path.stat().st_size == src_path.stat().st_size:
        return "skipped"
    shutil.copy2(src_path, dst_path)
    return "copied"


def drain_done(pending, stats, block=False):
    if not pending:
        return pending
    done, pending = wait(pending, return_when=ALL_COMPLETED if block else FIRST_COMPLETED)
    for future in done:
        stats[future.result()] += 1
    return pending


def build_item(item, new_id, rel_image):
    out = {
        "id": new_id,
        "image": rel_image,
        "text": item.get("text", ""),
        "fake_cls": item.get("fake_cls", ""),
        "fake_image_box": item.get("fake_image_box", []),
    }
    if "Internvl_out_think" in item:
        out["Internvl_out_think"] = item["Internvl_out_think"]
    return out


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    domain_counters = {spec["domain"]: 0 for spec in DATASET_SPECS}
    manifest = {
        "name": "ROM",
        "description": "Reasoning-enhanced analysis for Omnibus Manipulation dataset",
        "splits": {},
    }
    copy_stats = {"copied": 0, "skipped": 0}
    pending = set()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for spec in DATASET_SPECS:
            split = spec["split"]
            domain = spec["domain"]
            domain_dir = output_root / split / domain
            image_dir = domain_dir / "images"
            meta_path = domain_dir / "meta.json"
            image_dir.mkdir(parents=True, exist_ok=True)

            exported = []
            for source_json in spec["paths"]:
                with open(source_json, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for item in data:
                    domain_counters[domain] += 1
                    new_id = f"ROM{domain}{domain_counters[domain]:08d}"
                    source_image = Path(item["image"])
                    suffix = source_image.suffix.lower() or ".jpg"
                    rel_image = f"{split}/{domain}/images/{new_id}{suffix}"
                    dst_image = output_root / rel_image
                    exported.append(build_item(item, new_id, rel_image))

                    if not args.dry_run:
                        pending.add(
                            executor.submit(copy_image, source_image, dst_image, args.overwrite)
                        )
                        if len(pending) >= max(args.workers * 8, 1):
                            pending = drain_done(pending, copy_stats)

                    total_seen = sum(s["num_samples"] for s in manifest["splits"].values()) + len(exported)
                    if args.log_every and total_seen % args.log_every == 0:
                        print(
                            f"[export] processed={total_seen} copied={copy_stats['copied']} "
                            f"skipped={copy_stats['skipped']} current={split}/{domain}",
                            flush=True,
                        )

            if not args.dry_run:
                pending = drain_done(pending, copy_stats, block=True)
                with meta_path.open("w", encoding="utf-8") as f:
                    json.dump(exported, f, ensure_ascii=False, separators=(",", ":"))

            key = f"{split}/{domain}"
            manifest["splits"][key] = {
                "num_samples": len(exported),
                "meta": f"{split}/{domain}/meta.json",
                "image_dir": f"{split}/{domain}/images",
            }
            print(f"[export] wrote {key}: {len(exported)} samples", flush=True)

    manifest["copy_stats"] = copy_stats
    manifest["total_samples"] = sum(split["num_samples"] for split in manifest["splits"].values())
    manifest_path = output_root / "manifest.json"
    if not args.dry_run:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[export] done: {manifest['total_samples']} samples -> {output_root}", flush=True)
    print(f"[export] copy stats: {copy_stats}", flush=True)


if __name__ == "__main__":
    main()
