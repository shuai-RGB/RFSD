"""RFSD configuration, dataset presets, CLI parsing, and executable entry."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


# When this file is executed inside ``rfsd``, re-enter it as a package module.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if __name__ == "__main__":
        from rfsd.main import main as package_main

        package_main()
        raise SystemExit(0)


BEST_PARAMS_PATH = Path(__file__).with_name("best_params.yaml")


def _parse_yaml_scalar(value: str):
    """Parse the scalar types used in the flat RFSD preset YAML file."""

    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_dataset_presets(path: Path = BEST_PARAMS_PATH) -> dict:
    """Load the two-level mapping stored in ``best_params.yaml``."""

    if not path.is_file():
        raise FileNotFoundError(f"RFSD best-parameter file not found: {path}")
    presets = {}
    current_dataset = None
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw_line[0].isspace():
            if not stripped.endswith(":"):
                raise ValueError(
                    f"Invalid dataset entry at {path}:{line_number}: {stripped}"
                )
            current_dataset = stripped[:-1].strip()
            if not current_dataset:
                raise ValueError(f"Empty dataset name at {path}:{line_number}")
            presets[current_dataset] = {}
            continue
        if current_dataset is None or ":" not in stripped:
            raise ValueError(
                f"Invalid parameter entry at {path}:{line_number}: {stripped}"
            )
        parameter_name, raw_value = stripped.split(":", 1)
        parameter_name = parameter_name.strip()
        if not parameter_name or not raw_value.strip():
            raise ValueError(
                f"Invalid parameter entry at {path}:{line_number}: {stripped}"
            )
        presets[current_dataset][parameter_name] = _parse_yaml_scalar(raw_value)
    if not presets:
        raise ValueError(f"No dataset presets found in {path}")
    return presets


@dataclass
class RFSDConfig:
    """All parameters needed to prepare data, build RFSD, and train it."""

    seed: int = 42
    dataset: str = "amazon"
    eval_k: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 2000
    device: str = "cuda:0"
    num_clusters: int = 2
    data_dir: str = "data/amazon"
    interaction_layers: int = 1
    text_embedding_dim: int = 1536
    id_embedding_dim: int = 256
    item_graph_layers: int = 1
    user_graph_layers: int = 1
    contrastive_weight: float = 1.0
    user_top_k: int = 20
    item_top_k: int = 20
    num_negatives: int = 5
    regularization_weight: float = 1e-4
    spectral_dim: int = 128
    cluster_temperature: float = 0.6
    contrastive_temperature: float = 0.2
    hard_negative_margin: float = 0.3
    hard_negative_scale: float = 5.0
    noise_rate: float = 0.0
    early_stop: bool = False
    patience: int = 50
    min_delta: float = 1e-5

    @classmethod
    def from_namespace(cls, namespace: Any) -> "RFSDConfig":
        """Create a config from either the new or legacy argument names."""

        aliases = {
            "eval_k": "eva_k",
            "learning_rate": "lr",
            "max_epochs": "max_epoch",
            "num_clusters": "cluster",
            "data_dir": "folder_name",
            "interaction_layers": "num_layers",
            "text_embedding_dim": "txt_ebd_dim",
            "id_embedding_dim": "id_ebd_dim",
            "item_graph_layers": "n_ii_layers",
            "user_graph_layers": "n_uu_layers",
            "contrastive_weight": "lambda1",
            "user_top_k": "u_topk",
            "item_top_k": "i_topk",
            "num_negatives": "num_neg",
            "regularization_weight": "reg_weight",
            "spectral_dim": "spectual_dim",
            "cluster_temperature": "temperature",
            "contrastive_temperature": "temper_cl",
            "hard_negative_margin": "gamma",
            "hard_negative_scale": "beta",
        }
        values = {}
        for field in fields(cls):
            if hasattr(namespace, field.name):
                values[field.name] = getattr(namespace, field.name)
            elif field.name in aliases and hasattr(namespace, aliases[field.name]):
                values[field.name] = getattr(namespace, aliases[field.name])
        return cls(**values)


def _load_validated_presets() -> dict:
    presets = load_dataset_presets()
    config_fields = {field.name for field in fields(RFSDConfig)}
    for dataset, parameters in presets.items():
        unknown_parameters = set(parameters) - config_fields
        if unknown_parameters:
            names = ", ".join(sorted(unknown_parameters))
            raise ValueError(
                f"Unknown RFSD parameters for dataset {dataset!r}: {names}"
            )
        if "data_dir" not in parameters:
            raise ValueError(
                f"Dataset {dataset!r} must define data_dir in {BEST_PARAMS_PATH}"
            )
    return presets


DATASET_PRESETS = _load_validated_presets()


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the RFSD CLI while accepting the original option names."""

    parser = argparse.ArgumentParser(
        description="Train the RFSD recommendation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data",
        dest="dataset",
        choices=tuple(DATASET_PRESETS),
        default="amazon",
        help="dataset preset",
    )
    parser.add_argument("--eval-k", "--eva_k", dest="eval_k", type=int, default=20)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-2)
    parser.add_argument("--max-epochs", "--max_epoch", dest="max_epochs", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-clusters", "--cluster", dest="num_clusters", type=int, default=2)
    parser.add_argument("--data-dir", "--folder_name", dest="data_dir", type=str, default="data/amazon")
    parser.add_argument("--interaction-layers", "--num_layers", dest="interaction_layers", type=int, default=1)
    parser.add_argument(
        "--text-embedding-dim", "--txt_ebd_dim", dest="text_embedding_dim", type=int, default=1536
    )
    parser.add_argument("--id-embedding-dim", "--id_ebd_dim", dest="id_embedding_dim", type=int, default=256)
    parser.add_argument(
        "--item-graph-layers", "--n_ii_layers", dest="item_graph_layers", type=int, default=1
    )
    parser.add_argument(
        "--user-graph-layers", "--n_uu_layers", dest="user_graph_layers", type=int, default=1
    )
    parser.add_argument("--contrastive-weight", "--lambda1", dest="contrastive_weight", type=float, default=1.0)
    parser.add_argument("--user-top-k", "--u_topk", dest="user_top_k", type=int, default=20)
    parser.add_argument("--item-top-k", "--i_topk", dest="item_top_k", type=int, default=20)
    parser.add_argument("--num-negatives", "--num_neg", dest="num_negatives", type=int, default=5)
    parser.add_argument(
        "--regularization-weight", "--reg_weight", dest="regularization_weight", type=float, default=1e-4
    )
    parser.add_argument("--spectral-dim", "--spectual_dim", dest="spectral_dim", type=int, default=128)
    parser.add_argument(
        "--cluster-temperature", "--temperature", dest="cluster_temperature", type=float, default=0.6
    )
    parser.add_argument(
        "--contrastive-temperature", "--temper_cl", dest="contrastive_temperature", type=float, default=0.2
    )
    parser.add_argument("--hard-negative-margin", "--gamma", dest="hard_negative_margin", type=float, default=0.3)
    parser.add_argument("--hard-negative-scale", "--beta", dest="hard_negative_scale", type=float, default=5.0)
    parser.add_argument("--noise-rate", "--noise_rate", dest="noise_rate", type=float, default=0.0)
    parser.add_argument("--early-stop", "--early_stop", dest="early_stop", action="store_true")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", "--min_delta", dest="min_delta", type=float, default=1e-5)
    return parser


def parse_config(args: list[str] | None = None) -> RFSDConfig:
    """Parse arguments, apply a dataset preset, then explicit overrides."""

    raw_args = list(sys.argv[1:] if args is None else args)
    parser = build_argument_parser()
    namespace = parser.parse_args(raw_args)
    config = RFSDConfig.from_namespace(namespace)

    for field_name, value in DATASET_PRESETS[namespace.dataset].items():
        setattr(config, field_name, value)

    explicit_destinations = set()
    for action in parser._actions:
        for option in action.option_strings:
            if any(
                argument == option or argument.startswith(option + "=")
                for argument in raw_args
            ):
                explicit_destinations.add(action.dest)
                break
    for destination in explicit_destinations:
        if hasattr(config, destination) and hasattr(namespace, destination):
            setattr(config, destination, getattr(namespace, destination))
    return config


def main() -> None:
    config = parse_config()
    from rfsd.trainer import train_rfsd

    train_rfsd(config)
