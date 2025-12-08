import yaml
import argparse
import os
from dataclasses import dataclass
from typing import List

# MODEL OPTIONS
@dataclass
class ModelOpts:
    batch_norm: bool
    inchannels: int
    classes: int
    lr: float
    weights_decay: float

# TRAINING OPTIONS
@dataclass
class TrainingOpts:
    run_name: str
    num_epochs: int
    batch_size: int

# METRICS OPTIONS
@dataclass
class MetricThreshold:
    threshold: float

@dataclass
class MetricsOpts:
    IoU: MetricThreshold
    pixel_acc: MetricThreshold
    precision: MetricThreshold
    recall: MetricThreshold
    dice: MetricThreshold

# PATHS (TRAINING + INFERENCE)
@dataclass
class TrainingPaths:
    config_file: str
    dataset_path: str
    save_path: str
    train_csv: str
    val_csv: str
    train_img_dir: str
    val_img_dir: str

@dataclass
class InferencePaths:
    config_file: str
    dataset_path: str
    test_csv: str
    train_img_dir: str


@dataclass
class Paths:
    training: TrainingPaths
    inference: InferencePaths

@dataclass
class Config:
    model_opts: ModelOpts
    training_opts: TrainingOpts
    metrics_opts: MetricsOpts
    paths: Paths


def load_config(path: str = "config.yaml") -> Config:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    return Config(
        model_opts=ModelOpts(**raw_config["model_opts"]),
        training_opts=TrainingOpts(**raw_config["training_opts"]),
        metrics_opts=MetricsOpts(
            IoU=MetricThreshold(**raw_config["metrics_opts"]["IoU"]),
            pixel_acc=MetricThreshold(**raw_config["metrics_opts"]["pixel_acc"]),
            precision=MetricThreshold(**raw_config["metrics_opts"]["precision"]),
            recall=MetricThreshold(**raw_config["metrics_opts"]["recall"]),
            dice=MetricThreshold(**raw_config["metrics_opts"]["dice"]),
        ),
        paths=Paths(
            training=TrainingPaths(**raw_config["paths"]["training"]),
            inference=InferencePaths(**raw_config["paths"]["inference"]),
        ),
    )

def get_config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    return parser
