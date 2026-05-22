import argparse
from pathlib import Path
import torch
from src.config import Config
from src.data_utils import (
    convert_indices_to_techniques_batch,
    get_unique_tactics_from_techniques,
    map_techniques_to_tactics,
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
from src.visualize import plot_confusion_matrix, plot_roc_curve, plot_topk_predictions, plot_training_loss


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
        metrics = evaluate_model(model, dataset["test_samples"], dataset["id2technique"], device=device)
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"AUC:       {metrics['auc']:.4f}")
        print("="*60)
        y_true_tech, y_pred_tech = convert_indices_to_techniques_batch(metrics['y_true'],metrics['y_pred'],dataset["id2technique"])
        # تبدیل تکنیک‌ها به تاکتیک‌ها
        y_true_tactics = map_techniques_to_tactics(
            y_true_tech,
            tactic_techniques_path="data/tactic_techniques.json"
        )

        # ─────────────────────────────────────────────────────────
        # تبدیل ایندکس‌ها به تکنیک‌ها
        # ─────────────────────────────────────────────────────────
        y_true_tech, y_pred_tech = convert_indices_to_techniques_batch(
            metrics['y_true'],
            metrics['y_pred'],
            dataset["id2technique"]
        )
    
        # ─────────────────────────────────────────────────────────
        # تبدیل تکنیک‌ها به تاکتیک‌ها
        # ─────────────────────────────────────────────────────────
        y_true_tactics = map_techniques_to_tactics(
            y_true_tech,
            tactic_techniques_path="data/tactic_techniques.json"
        )
    
        y_pred_tactics = map_techniques_to_tactics(
            y_pred_tech,
            tactic_techniques_path="data/tactic_techniques.json"
        )
        
        # ─────────────────────────────────────────────────────────
        # استخراج کلاس‌های یکتا (مرتب شده بر اساس tactic_order.json)
        # ─────────────────────────────────────────────────────────
        unique_tactics = get_unique_tactics_from_techniques(
            y_true_tech + y_pred_tech,
            tactic_techniques_path="data/tactic_techniques.json",
            tactic_order_path="data/tactic_order.json"
        )
        
        # ─────────────────────────────────────────────────────────
        # ساخت نگاشت تکنیک به تاکتیک
        # ─────────────────────────────────────────────────────────
        import json
        with open("data/tactic_techniques.json", "r") as f:
            tactic_techniques = json.load(f)
        
        technique_to_tactic_map = {}
        for tactic, techniques in tactic_techniques.items():
            for tech in techniques:
                technique_to_tactic_map[tech['external_id']] = tactic
    
        # ─────────────────────────────────────────────────────────
        # رسم نمودارها
        # ─────────────────────────────────────────────────────────
        
        # روش 1: رسم جداگانه
        plot_confusion_matrix(
            y_true_tactics, 
            y_pred_tactics,
            class_names=unique_tactics,
            save_path="plots/tactic_confusion_matrix.png"
        )
        
        plot_roc_curve(
            y_true_tactics,
            metrics['y_prob'],
            technique_to_tactic_map,
            dataset["id2technique"],
            class_names=unique_tactics,
            save_path="plots/tactic_roc_curve.png"
        )

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
