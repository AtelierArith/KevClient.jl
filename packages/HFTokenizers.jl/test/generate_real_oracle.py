"""Regenerate real_oracle.json from the pinned real tokenizers.

    python generate_real_oracle.py

Downloads each tokenizer.json at the revision recorded here (the same URL the
Julia test uses), and writes the token ids Hugging Face `tokenizers` produces
for STRINGS. The Julia test re-downloads the file at that revision and checks
HFTokenizer against these ids; it skips when the download fails.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")

TOKENIZERS = {
    "qwen": ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "roberta": ("FacebookAI/roberta-base", "e2da8e2f811d1448a5b465c236feacd80ffbac7b"),
}

STRINGS = [
    "Hello, world!",
    "The parcel arrived crushed and the customer wants a replacement.",
    "I see two charges for order #4411. Please refund one.",
    "  leading and   multiple spaces  ",
    "line1\nline2\r\nline3",
    "It's a test; can't, won't, shouldn't.",
    "caf\u00e9 na\u00efve r\u00e9sum\u00e9",
    "cafe\u0301 combining",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3068\u3001\u53e5\u8aad\u70b9\u3002",
    "emoji \U0001F600\U0001F44D\U0001F3FD test",
    "1234567890 3.14159 100%",
    "def foo(x): return x**2  # comment",
    "a<b>c&d",
    "mixed \u65e5\u672c\u8a9e and English 123",
    "",
    "\u0120\u0120",
]

ESCAPE_STRINGS = [
    "<|fim_prefix|>",
    "a <|fim_middle|> b",
    "<|box_start|>opt<|box_end|>",
    "<|fim_suffix|>",
]


def download(repo, revision):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, repo.replace("/", "--") + "@" + revision + ".json")
    if not os.path.isfile(path):
        import urllib.request
        url = "https://huggingface.co/%s/resolve/%s/tokenizer.json" % (repo, revision)
        print("downloading", url)
        urllib.request.urlretrieve(url, path)
    return path


def escape(text):
    import re
    return re.sub(r"<\|([A-Za-z0-9_]+)\|>", "<\u00a6\\1\u00a6>", text)


def main():
    from tokenizers import Tokenizer

    oracle = {"strings": STRINGS, "tokenizers": {}}
    for name, (repo, revision) in TOKENIZERS.items():
        path = download(repo, revision)
        tok = Tokenizer.from_file(path)
        entry = {
            "repo": repo,
            "revision": revision,
            "ids": [tok.encode(s, add_special_tokens=False).ids for s in STRINGS],
        }
        if name == "qwen":
            entry["escaped_strings"] = [escape(s) for s in ESCAPE_STRINGS]
            entry["escaped_ids"] = [tok.encode(escape(s), add_special_tokens=False).ids for s in ESCAPE_STRINGS]
        oracle["tokenizers"][name] = entry
    with open(os.path.join(HERE, "real_oracle.json"), "w", encoding="utf-8") as handle:
        json.dump(oracle, handle, ensure_ascii=False, indent=1)
    print("wrote real_oracle.json")


if __name__ == "__main__":
    main()
