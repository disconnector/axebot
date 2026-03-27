#!/usr/bin/env python3
"""
Fine-tune all-MiniLM-L6-v2 on Axe-Fx III domain pairs.

Uses MultipleNegativesRankingLoss — the standard loss for embedding fine-tuning.
Each batch of (anchor, positive) pairs treats other positives in the batch as
negatives automatically. No explicit negative mining required.

Input:  ~/rag/training_pairs.jsonl
Output: ~/rag/models/axefx-embed/
"""

import json
import argparse
import random
from pathlib import Path
from datetime import datetime

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from datasets import Dataset

PAIRS_FILE  = Path.home() / "rag" / "training_pairs.jsonl"
MODEL_OUT   = Path.home() / "rag" / "models" / "axefx-embed"
BASE_MODEL  = "all-MiniLM-L6-v2"


def load_pairs(path, val_split=0.05):
    """Load JSONL pairs and split into train/val sets."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    random.shuffle(pairs)
    n_val   = max(50, int(len(pairs) * val_split))
    val     = pairs[:n_val]
    train   = pairs[n_val:]

    print(f"  Loaded {len(pairs)} pairs → {len(train)} train / {len(val)} val")
    return train, val


def make_dataset(pairs):
    """Convert list of dicts to HuggingFace Dataset."""
    return Dataset.from_dict({
        "anchor":   [p["anchor"]   for p in pairs],
        "positive": [p["positive"] for p in pairs],
    })


def make_evaluator(val_pairs, model):
    """
    Build an InformationRetrievalEvaluator from val pairs.
    Treats each val anchor as a query and its positive as the relevant doc.
    """
    queries   = {str(i): p["anchor"]   for i, p in enumerate(val_pairs)}
    corpus    = {str(i): p["positive"] for i, p in enumerate(val_pairs)}
    relevant  = {str(i): {str(i)}      for i in range(len(val_pairs))}

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        name="axefx-val",
        show_progress_bar=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune embedding model on Axe-Fx III pairs")
    parser.add_argument("--epochs",     type=int,   default=3,    help="Training epochs")
    parser.add_argument("--batch-size", type=int,   default=32,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--pairs-file", type=str,   default=str(PAIRS_FILE), help="Path to training pairs JSONL")
    args = parser.parse_args()

    start = datetime.now()
    print(f"[{start}] Starting fine-tuning")
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  Pairs file:  {args.pairs_file}")
    print(f"  Output:      {MODEL_OUT}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading pairs...")
    train_pairs, val_pairs = load_pairs(args.pairs_file)
    train_dataset = make_dataset(train_pairs)

    # ── Load base model ────────────────────────────────────────────────────────
    print(f"\nLoading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)

    # ── Loss function ──────────────────────────────────────────────────────────
    # MultipleNegativesRankingLoss:
    # For each (anchor, positive) pair in the batch, all other positives
    # become implicit negatives. Large batch size = more negatives = better training.
    loss = MultipleNegativesRankingLoss(model)

    # ── Evaluator ──────────────────────────────────────────────────────────────
    print("Building evaluator...")
    evaluator = make_evaluator(val_pairs, model)

    # ── Training arguments ────────────────────────────────────────────────────
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(MODEL_OUT),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,           # warm up LR for first 10% of steps
        lr_scheduler_type="cosine", # cosine decay after warmup
        bf16=False,                 # CPU training — no bfloat16
        fp16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="axefx-val_cosine_ndcg@10",
        logging_steps=50,
        report_to="none",           # no wandb/tensorboard
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()

    # ── Save final model ───────────────────────────────────────────────────────
    final_path = MODEL_OUT / "final"
    model.save(str(final_path))

    elapsed = datetime.now() - start
    print(f"\nTraining complete in {elapsed}")
    print(f"  Model saved to: {final_path}")
    print(f"\nNext steps:")
    print(f"  1. Update EMBED_MODEL in ingest.py and query.py to: {final_path}")
    print(f"  2. Re-ingest corpus: python ingest.py --force")
    print(f"  3. Test: python query.py 'your question'")


if __name__ == "__main__":
    main()
