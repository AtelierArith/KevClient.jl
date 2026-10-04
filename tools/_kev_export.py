"""Stage a Kev checkpoint into a bundle KevClient.jl can load without Python.

Reads ``head.pt`` (a torch pickle) and writes:

    <out>/kev_meta.json             # Meta fields, plus backbone/backbone_revision
    <out>/pointer_head.safetensors  # head.pt["head"]: head.q.weight, head.q.bias, ...
    <out>/tokenizer.json ...        # copied when present (for future tokenization)

A full-weight checkpoint records the original checkpoint as ``backbone`` so its
tens of GB are never copied; a LoRA checkpoint keeps ``base``/``base_revision``
and the backbone is resolved from the Hugging Face cache at load time.

Invoked by ``export_kev_checkpoint.jl`` through PythonCall.jl; torch is imported
lazily so the module imports (and the test suite compiles) without it.
"""

import argparse
import json
import os
import shutil
import sys

# Fields KevClient's KevMeta reads; anything else in head.pt is preserved too.
KNOWN = (
    "base",
    "base_revision",
    "lora",
    "head_dim",
    "option_isolation",
    "special_embeddings",
    "weights_dtype",
    "temperature",
    "holdout",
    "weights",
    "backbone",
    "backbone_revision",
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def split_revision(source, revision):
    """Accept ``org/model@rev`` as an alternative to ``--revision``."""
    if revision or os.path.isdir(source) or "@" not in source:
        return source, revision
    repo, _, tail = source.rpartition("@")
    return repo, tail or None


def resolve(source, revision):
    """Return a local directory for a checkpoint path or Hub repo id."""
    if os.path.isdir(source):
        return os.path.abspath(source)
    from huggingface_hub import snapshot_download

    return snapshot_download(
        source,
        revision=revision or None,
        allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage a Kev checkpoint for KevClient.jl.")
    parser.add_argument("source", help="local checkpoint directory, or org/model[@revision]")
    parser.add_argument("--out", required=True, help="bundle output directory")
    parser.add_argument("--revision", default=None, help="Hub revision (branch, tag or commit)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing bundle")
    args = parser.parse_args(argv)

    source, revision = split_revision(args.source, args.revision)
    is_local_source = os.path.isdir(source)
    out = os.path.abspath(os.path.expanduser(args.out))
    head_path = None
    if is_local_source:
        candidate = os.path.join(os.path.abspath(source), "head.pt")
        if os.path.isfile(candidate):
            head_path = candidate

    checkpoint = resolve(source, revision) if head_path is None else os.path.dirname(head_path)
    if head_path is None:
        head_path = os.path.join(checkpoint, "head.pt")
    if not os.path.isfile(head_path):
        raise SystemExit(f"no head.pt under {checkpoint}")

    if os.path.exists(out) and os.listdir(out) and not args.force:
        raise SystemExit(f"{out} already exists; pass --force to overwrite")

    import torch
    from safetensors.torch import save_file

    meta = torch.load(head_path, map_location="cpu")
    if "head" not in meta or not meta["head"]:
        raise SystemExit(f"{head_path} has no 'head' state dict")
    head = {f"head.{name}": tensor.detach().cpu().contiguous() for name, tensor in meta["head"].items()}
    for required in ("head.q.weight", "head.q.bias", "head.k.weight", "head.k.bias"):
        if required not in head:
            raise SystemExit(f"{head_path} is missing {required}")

    os.makedirs(out, exist_ok=True)
    save_file(head, os.path.join(out, "pointer_head.safetensors"))

    record = {key: meta[key] for key in KNOWN if key in meta}
    for key, value in meta.items():  # keep training args, temperature_fit, ...
        record.setdefault(key, value)
    record.pop("head", None)
    if str(record.get("weights", "lora")) == "full":
        # Point at the original checkpoint; never copy the backbone shards.
        record["backbone"] = os.path.abspath(source) if is_local_source else source
        if not is_local_source:
            record["backbone_revision"] = revision
    else:
        record.setdefault("backbone", None)
        record.setdefault("backbone_revision", None)

    with open(os.path.join(out, "kev_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")

    for name in TOKENIZER_FILES:
        src = os.path.join(checkpoint, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out, name))

    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
