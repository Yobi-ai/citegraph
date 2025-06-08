import cProfile
import csv
import logging
import os
import pstats
import random
import sys
from pathlib import Path
from typing import List, Tuple

import hydra
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from rich import print

from ...utils.monitor import log_system_metrics
from .dataloader import Dataset
from .model import GCN

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)


logger = logging.getLogger(__name__)
log_dir = os.path.join(os.path.normpath(os.getcwd()), "logs")
if "logs" not in os.listdir():
    os.mkdir("logs")
FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
formatter = logging.Formatter(FORMAT)
file_handler = logging.FileHandler(f"{log_dir}/citegraph_train.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.propagate = False

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("citegraph")


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self._epochs = cfg.epochs
        self._model_save_freq = cfg.model_save_freq
        self._print_stats_freq = cfg.print_stats_freq

        self._embeddings_over_time: List[torch.Tensor] = []

        self._train_loss_over_time: List[float] = []
        self._val_loss_over_time: List[float] = []
        self._train_acc_over_time: List[float] = []
        self._val_acc_over_time: List[float] = []
        self._file = open("training_results.csv", "w")
        self._csv_writer = csv.writer(self._file)

        self.__intialize_objects(cfg.data_path, cfg.hidden_dim, cfg.lr)

    def __intialize_objects(self, data_path: str, hidden_dim: int, lr: float) -> None:
        dataloader = Dataset()
        dataset = dataloader.load_cora(data_path)
        self.data = dataset[0]
        self.model = GCN(dataset.num_node_features, hidden_dim, dataset.num_classes)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=5e-4
        )

        self.__move_to_device()

    def __move_to_device(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)
        self.data.to(self._device)
        logger.info(f"Device: {self._device}")
        print(f"Device: [cyan]{self._device}[/cyan]")

    def __train_epoch(self, model: GCN, optimizer: torch.optim.Optimizer) -> float:
        model.train()
        optimizer.zero_grad()
        out = model(self.data.x, self.data.edge_index)
        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def __validate_epoch(self, model: GCN) -> Tuple[float, List[float], torch.Tensor]:
        model.eval()
        with torch.no_grad():
            out = model(self.data.x, self.data.edge_index)
        val_loss = F.nll_loss(
            out[self.data.val_mask], self.data.y[self.data.val_mask]
        ).item()
        pred = out.argmax(dim=1)
        embeddings = out.cpu().detach()
        accs = []
        for mask in [self.data.train_mask, self.data.val_mask]:
            correct = pred[mask].eq(self.data.y[mask]).sum().item()
            accs.append(correct / mask.sum().item())
        return val_loss, accs, embeddings

    def get_full_training_results(
        self,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        return (
            self._train_loss_over_time,
            self._train_acc_over_time,
            self._val_loss_over_time,
            self._val_acc_over_time,
        )

    def train(self) -> None:
        print("Starting Training")
        logger.info("Starting Training")

        # Create a Profile object
        profiler = cProfile.Profile()
        profiler.enable()

        with mlflow.start_run(run_name="gcn-train") as run:
            try:
                for epoch in range(1, self._epochs + 1):
                    train_loss = self.__train_epoch(self.model, self.optimizer)
                    val_loss, acc, embeddings = self.__validate_epoch(self.model)
                    train_acc, val_acc = acc

                    self._train_loss_over_time.append(train_loss)
                    self._val_loss_over_time.append(val_loss)
                    self._train_acc_over_time.append(train_acc)
                    self._val_acc_over_time.append(val_acc)
                    self._csv_writer.writerow(
                        [epoch, train_loss, train_acc, val_loss, val_acc]
                    )

                    mlflow.log_metric("train_loss", train_loss)
                    mlflow.log_metric("train_accuracy", train_acc)
                    mlflow.log_metric("val_loss", val_loss)
                    mlflow.log_metric("val_accuracy", val_acc)

                    log_system_metrics(epoch)

                    if epoch % self._print_stats_freq == 0:
                        print(
                            f"[yellow]Epoch: {epoch:03d}, "
                            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}[/yellow]"
                        )
                        logger.info(
                            f"Epoch: {epoch:03d}, "
                            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                        )

                    if epoch % self._model_save_freq == 0:
                        print(f"Saving model at [green]epoch {epoch}[/green]")
                        logger.info(f"Saving model at epoch {epoch}")
                        torch.save(self.model, f"./model_{epoch}.pth")
                        mlflow.pytorch.log_model(self.model, f"model_{epoch}")

                print("[bold green]Training Completed Successfully![/bold green]")
                logger.info("Training Completed Successfully!")
                print(f"mlflow run ID: {run.info.run_id}")

            except KeyboardInterrupt:
                print("Keyboard Interrupt\n[bold red]Stopping Training![/bold red]")
                logger.info("Keyboard Interrupt\nStopping Training!")
                print(f"mlflow run ID: {run.info.run_id}")

            finally:
                # Disable profiler and save results
                profiler.disable()
                stats = pstats.Stats(profiler)
                stats.sort_stats("cumulative")
                stats.dump_stats("training_profile.prof")
                print(
                    "[bold green]Profiling results saved to training_profile.prof[/bold green]"
                )
                logger.info("Profiling results saved to training_profile.prof")

                # Print top 20 time-consuming functions
                print("\n[bold cyan]Top 20 Time-Consuming Functions:[/bold cyan]")
                stats.strip_dirs().sort_stats("cumulative").print_stats(20)

        self.__cleanup()

    def __cleanup(self) -> None:
        self._file.close()


@hydra.main(version_base=None, config_path="confs/train", config_name="training_conf")
def main(cfg: DictConfig) -> None:
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    print(f"[yellow]Configuration:\n{OmegaConf.to_yaml(cfg)}[/yellow]")
    trainer_obj = Trainer(cfg)
    trainer_obj.train()


if __name__ == "__main__":
    main()
