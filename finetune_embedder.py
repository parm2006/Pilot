#!/usr/bin/env python3
"""
finetune_embedder.py - teach a small local embedding model YOUR voice commands
===============================================================================

Goal: spoken paraphrases ("grab everything") should land right next to the canonical phrase
("select all") in embedding space, and unrelated speech should land far from every command.

Training data (all from files next to this script):
  * commands.json   - every phrase and every "aliases" entry
  * paraphrases.json - {"select all": ["grab everything", ...]}  (add many more, in your own words)
Each (paraphrase -> canonical phrase) pair is one training example, trained with
MultipleNegativesRankingLoss: the other commands in the batch act as negatives.

Setup (a GPU is nice but not needed for a few hundred pairs):
    pip install sentence-transformers datasets

Run:
    python finetune_embedder.py                      # base all-MiniLM-L6-v2 -> ./pilot-embed
    python finetune_embedder.py --epochs 8 --base sentence-transformers/all-MiniLM-L6-v2
    python finetune_embedder.py --push-to-hub YOURNAME/pilot-voice-embed   # publish the weights (you must be logged in: `hf auth login`)

Use the result in pilot.py:
    python pilot.py --embed-backend st --embed-model ./pilot-embed

It prints top-1 accuracy on HELD-OUT paraphrases before and after, so you can see whether it helped.
With only a handful of paraphrases per command it can overfit - keep the held-out score as your judge.
Note: Ollama cannot load these weights directly; sentence-transformers is the supported path here.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def norm(s: str) -> str:
    return " ".join(s.lower().replace("[unk]", " ").split())


def load_pairs(held_out: float, seed: int):
    commands = json.loads((HERE / "commands.json").read_text(encoding="utf-8"))["commands"]
    para_path = HERE / "paraphrases.json"
    para = json.loads(para_path.read_text(encoding="utf-8")) if para_path.exists() else {}
    rng = random.Random(seed)
    train, test = [], []
    canon = sorted(norm(k) for k in commands)
    for phrase, action in commands.items():
        c = norm(phrase)
        for alias in action.get("aliases", []):          # aliases always train: they are your declared synonyms
            train.append((norm(alias), c))
        train.append((c, c))
    for phrase, options in para.items():
        if phrase.startswith("_") or norm(phrase) not in canon:
            continue
        options = list(options)
        rng.shuffle(options)
        k = max(1, round(len(options) * held_out)) if len(options) >= 4 else 0
        test += [(norm(o), norm(phrase)) for o in options[:k]]
        train += [(norm(o), norm(phrase)) for o in options[k:]]
    return train, test, canon


def accuracy(model, pairs, canon) -> float:
    if not pairs:
        return float("nan")
    C = model.encode(canon, normalize_embeddings=True)
    Q = model.encode([q for q, _ in pairs], normalize_embeddings=True)
    pred = (Q @ C.T).argmax(1)
    return sum(canon[p] == t for p, (_, t) in zip(pred, pairs)) / len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default=str(HERE / "pilot-embed"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--held-out", type=float, default=0.25, help="fraction of paraphrases kept for evaluation")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--push-to-hub", default=None, metavar="USER/REPO")
    a = ap.parse_args()

    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    train, test, canon = load_pairs(a.held_out, a.seed)
    print(f"{len(train)} training pairs, {len(test)} held-out paraphrases, {len(canon)} commands")
    model = SentenceTransformer(a.base)
    print(f"held-out top-1 accuracy BEFORE: {accuracy(model, test, canon):.1%}")

    ds = Dataset.from_dict({"anchor": [x for x, _ in train], "positive": [y for _, y in train]})
    args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(a.out) / "_runs"), num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch, learning_rate=a.lr, warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,        # same command twice in a batch would be a false negative
        logging_steps=10, save_strategy="no", report_to="none", seed=a.seed)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                               loss=losses.MultipleNegativesRankingLoss(model)).train()

    print(f"held-out top-1 accuracy AFTER:  {accuracy(model, test, canon):.1%}")
    model.save(a.out)
    print(f"saved to {a.out}  ->  python pilot.py --embed-backend st --embed-model {a.out}")
    if a.push_to_hub:
        model.push_to_hub(a.push_to_hub)
        print(f"pushed to https://huggingface.co/{a.push_to_hub}")


if __name__ == "__main__":
    main()
