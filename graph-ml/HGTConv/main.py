import argparse
from pathlib import Path

import torch

from src.config import Config
from src.data_utils import (
    set_seed,
    load_json_data,
    extract_sequences,
    build_vocab,
    build_sequence_chain_seq,
    sequence_to_heterodata,
    split_chain_seq,
)
from src.model_EA_HGT import EAHGT
from src.train import train_model
from src.evaluate import evaluate_model
from src.inference import predict_next_techniques
from src.visualize import plot_topk_predictions, plot_training_loss


def build_graph(sample, technique2id):
    graph = sequence_to_heterodata(sample["input_seq"], technique2id)
    sample["graph"] = graph
    return sample


def prepare_dataset(config: Config):
    raw_data = load_json_data(config.DATA_PATH)
    records = extract_sequences(raw_data)
    technique2id, id2technique, tactic2id, id2tactic = build_vocab(records)
    samples = build_sequence_chain_seq(records, technique2id, config.MAX_SEQ_LEN)

    samples = [build_graph(sample, technique2id) for sample in samples]
    train_samples, val_samples, test_samples = split_chain_seq(
        samples,
        test_ratio=config.TEST_RATIO,
        val_ratio=config.VAL_RATIO,
        seed=config.SEED
    )

    return {
        "records": records,
        "samples": samples,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "technique2id": technique2id,
        "id2technique": id2technique,
        "tactic2id": tactic2id,
        "id2tactic": id2tactic,
    }


def main():
    parser = argparse.ArgumentParser(description="APT Next Technique Prediction")
    parser.add_argument("--mode", choices=["train", "eval", "infer"], required=True)
    parser.add_argument(
        "--input_seq",
        type=str,
        default="T1071.001,T1102,T1132.001",
        help="Comma-separated technique sequence for inference"
    )
    args = parser.parse_args()

    config = Config()
    set_seed(config.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = prepare_dataset(config)

    model = EAHGT(
        num_techniques=len(dataset["technique2id"]),
        hidden_dim=config.HIDDEN_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    )

    if args.mode == "train":
        losses = train_model(model, dataset["train_samples"], dataset["technique2id"], config, device=device)
        # torch.save(
        #     {
        #         "model_state_dict": model.state_dict(),
        #         "technique2id": dataset["technique2id"],
        #         "id2technique": dataset["id2technique"],
        #     },
        #     config.CHECKPOINT_PATH,
        # )
        print(f"Model saved to {config.CHECKPOINT_PATH}")
        plot_training_loss(losses, save_path="plots/loss_curve.png")

    elif args.mode == "eval":
        if config.CHECKPOINT_PATH.exists():
            ckpt = torch.load(config.CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print("Checkpoint loaded.")
        metrics = evaluate_model(model, dataset["test_samples"], device=device)
        print(metrics)

    elif args.mode == "infer":
        if config.CHECKPOINT_PATH.exists():
            ckpt = torch.load(config.CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print("Checkpoint loaded.")
        else:
            print("Warning: checkpoint not found, using untrained model.")

        input_seq = [x.strip() for x in args.input_seq.split(",") if x.strip()]
        results = predict_next_techniques(
            model=model,
            input_seq=input_seq,
            technique2id=dataset["technique2id"],
            id2technique=dataset["id2technique"],
            data_builder=lambda seq: sequence_to_heterodata(seq, dataset["technique2id"]),
            top_k=config.TOP_K,
            device=device,
        )

        print("\nTop-K predicted next techniques:")
        for r in results:
            print(f"{r['technique_id']}  ->  {r['probability']:.4f}")

        plot_topk_predictions(results)


if __name__ == "__main__":
    main()
