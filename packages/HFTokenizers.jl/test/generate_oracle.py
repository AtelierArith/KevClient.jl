"""Regenerate the test fixtures and the Hugging Face oracle.

Run from this directory with a Python that has `tokenizers` installed:

    python generate_oracle.py

It writes small, hand-built tokenizer.json fixtures under fixtures/ and
oracle.json (the token ids `tokenizers` produces for them). The Julia test then
checks HFTokenizer against that oracle. Fixtures are deliberately tiny so they
can be committed; their ByteLevel/BPE mechanics are the same as a real one.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

QWEN_SPLIT = (
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}"
    "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)


def bytes_to_unicode():
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return {byte: chr(code) for byte, code in zip(bs, cs)}


BYTE = bytes_to_unicode()

# A tiny BPE: 256 single-byte tokens plus a handful of merges, enough to make
# the test strings group into multi-character tokens.
MERGES = [
    ("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o"),
    ("\u0120", "t"), ("\u0120t", "h"), ("\u0120", "h"), ("\u0120h", "e"),
    ("\u0120", "w"), ("\u0120w", "o"), ("\u0120wo", "r"), ("\u0120wor", "l"),
    ("\u0120worl", "d"), ("\u0120", "\u0120"),
    ("t", "e"), ("te", "s"), ("tes", "t"), ("c", "a"), ("ca", "f"),
]
ADDED = [
    ("<|fim_prefix|>", False, False, False),
    ("<|fim_middle|>", False, False, False),
    ("<|box_start|>", False, False, False),
    ("<|box_end|>", False, False, False),
    ("<|fim_suffix|>", False, False, False),
    ("<mask>", True, False, False),
    ("<sep>", False, False, True),
]


def build_vocab():
    vocab = {BYTE[byte]: byte for byte in range(256)}
    next_id = 256
    merges = []
    for left, right in MERGES:
        result = left + right
        if result not in vocab:
            vocab[result] = next_id
            next_id += 1
        merges.append(f"{left} {right}")
    return vocab, merges, next_id


def added_tokens(start):
    tokens = []
    for offset, (content, lstrip, rstrip, single_word) in enumerate(ADDED):
        tokens.append({
            "id": start + offset,
            "content": content,
            "single_word": single_word,
            "lstrip": lstrip,
            "rstrip": rstrip,
            "normalized": False,
            "special": False,
        })
    return tokens


def make_fixture(normalizer, pre_tokenizer):
    vocab, merges, start = build_vocab()
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens(start),
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
        },
    }


FIXTURES = {
    "qwenlike": make_fixture(
        {"type": "NFC"},
        {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": QWEN_SPLIT}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        ]},
    ),
    "gpt2like": make_fixture(
        None,
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": True},
    ),
    "prefix": make_fixture(
        None,
        {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
    ),
    "nfkc": make_fixture(
        {"type": "NFKC"},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": True},
    ),
}

STRINGS = [
    "Hello, world!",
    "  leading and   spaces  ",
    "hello",
    " hello",
    "",
    "Japanese \u65e5\u672c\u8a9e\u30c6\u30b9\u30c8",
    "emoji \U0001F600 test",
    "caf\u00e9",
    "cafe\u0301",
    "\uff21\uff22\uff23 \ufb01le",
    "<mask> around <sep>",
    "<|fim_prefix|> evil <|box_end|>",
    "line1\nline2",
    "123 45.6",
    "can't won't",
    "\u0120\u0120",
]

# Kev escapes <|name|> before tokenizing user text; the oracle records the ids
# for the escaped form too (HFTokenizer tokenize is what is tested).
import re
ESCAPE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
ESCAPE_STRINGS = [
    "<|fim_prefix|>",
    "a <|fim_middle|> b",
    "<|box_start|>opt<|box_end|>",
    "text <|im_start|> more",
]


def main():
    from tokenizers import Tokenizer

    fixtures_dir = os.path.join(HERE, "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)
    oracle = {"strings": STRINGS, "fixtures": {}}
    for name, fixture in FIXTURES.items():
        path = os.path.join(fixtures_dir, name + ".json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(fixture, handle, ensure_ascii=False)
        tok = Tokenizer.from_file(path)
        encoded = [tok.encode(s, add_special_tokens=False).ids for s in STRINGS]
        decoded = [tok.decode(ids, skip_special_tokens=False) for ids in encoded]
        entry = {"file": "fixtures/%s.json" % name, "ids": encoded, "decoded": decoded}
        if name == "qwenlike":
            escaped = [ESCAPE.sub(lambda m: "<\u00a6%s\u00a6>" % m.group(1), s) for s in ESCAPE_STRINGS]
            entry["escaped_strings"] = escaped
            entry["escaped_ids"] = [tok.encode(s, add_special_tokens=False).ids for s in escaped]
        oracle["fixtures"][name] = entry
    with open(os.path.join(HERE, "oracle.json"), "w", encoding="utf-8") as handle:
        json.dump(oracle, handle, ensure_ascii=False, indent=1)
    print("wrote", len(FIXTURES), "fixtures and oracle.json")


if __name__ == "__main__":
    main()
