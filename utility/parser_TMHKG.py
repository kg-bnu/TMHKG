import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run TMHKG")

    parser.add_argument(
        "--dataset",
        nargs="?",
        default="yelp",
        help="Choose a dataset: [yelp, nycrestraunt]",
    )
    parser.add_argument(
        "--data_path", nargs="?", default="datasets/", help="Input data path."
    )

    parser.add_argument("--dim", type=int, default=64, help="Embedding size.")
    parser.add_argument(
        "--context_hops", type=int, default=3, help="Number of GNN layers."
    )
    parser.add_argument(
        "--node_dropout_rate", type=float, default=0.5, help="Ratio of node dropout."
    )
    parser.add_argument(
        "--mess_dropout_rate", type=float, default=0.1, help="Ratio of message dropout."
    )
    parser.add_argument(
        "--max_history_len",
        type=int,
        default=20,
        help="Maximum length of user history for IEPE module.",
    )

    parser.add_argument(
        "--l2", type=float, default=1e-5, help="L2 regularization weight."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Weight for user representation alignment loss.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Weight for the overall representation consistency loss.",
    )

    parser.add_argument(
        "--epoch", type=int, default=100, help="Number of meta-training epochs."
    )
    parser.add_argument(
        "--meta_batch_size",
        type=int,
        default=16,
        help="Batch size for meta-training tasks.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for the main optimizer."
    )
    parser.add_argument(
        "--meta_update_lr",
        type=float,
        default=0.001,
        help="Learning rate for inner-loop updates in meta-learning.",
    )
    parser.add_argument(
        "--num_inner_update", type=int, default=1, help="Number of inner-loop updates."
    )

    parser.add_argument(
        "--fine_tune_epoch", type=int, default=100, help="Number of fine-tuning epochs."
    )
    parser.add_argument(
        "--fine_tune_batch_size",
        type=int,
        default=512,
        help="Batch size for fine-tuning.",
    )
    parser.add_argument(
        "--fine_tune_lr",
        type=float,
        default=0.0001,
        help="Learning rate for fine-tuning.",
    )

    parser.add_argument(
        "--test_batch_size", type=int, default=1024, help="Test batch size."
    )
    parser.add_argument(
        "--Ks",
        nargs="?",
        default="[10, 20]",
        help="K values for evaluation metrics (e.g., Recall@K, NDCG@K).",
    )
    parser.add_argument(
        "--cold_scenario",
        nargs="?",
        default="user_cold",
        help="Cold-start scenario to evaluate: [user_cold, poi_cold, user_poi_cold, warm_up]",
    )
    parser.add_argument(
        "--use_meta_model",
        action="store_true",
        help="Flag to load a pre-trained meta-model and skip meta-training.",
    )
    parser.add_argument(
        "--save", action="store_true", help="Flag to save the trained meta-model."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./model_para/",
        help="Output directory for saved models.",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--cuda", action="store_false", help="Disable CUDA training.")

    return parser.parse_args()
