import csv
import logging
import os
import sys
import cProfile
import pstats
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

import random

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from dataloader import Dataset
from model import GCN
from omegaconf import DictConfig, OmegaConf
from rich import print

from utils.monitor import log_system_metrics

logger = logging.getLogger(__name__)
log_dir = os.path.join(os.path.normpath(os.getcwd()), 'logs')
if "logs" not in os.listdir():
    os.mkdir("logs")
FORMAT = '%(asctime)s | %(levelname)s | %(message)s'
formatter = logging.Formatter(FORMAT)
file_handler = logging.FileHandler(f"{log_dir}/citegraph_train.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.propagate = False

class Trainer:
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self._epochs = cfg.epochs
        self._model_save_freq = cfg.model_save_freq
        self._print_stats_freq = cfg.print_stats_freq

        self._embeddings_over_time = []

        self._train_loss_over_time = []
        self._val_loss_over_time = []
        self._train_acc_over_time = []
        self._val_acc_over_time = []
        self._file = open('training_results.csv', 'w')
        self._csv_writer = csv.writer(self._file)

        self.__intialize_objects(cfg.data_path, cfg.hidden_dim, cfg.lr)

    def __intialize_objects(self, data_path, hidden_dim, lr):
        dataloader = Dataset()
        dataset = dataloader.load_cora(data_path)
        self.data = dataset[0]
        self.model = GCN(dataset.num_node_features, hidden_dim, dataset.num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)

        self.__move_to_device()

    def __move_to_device(self):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self._device)
        self.data.to(self._device)
        logger.info(f"Device: {self._device}")
        print(f"Device: [cyan]{self._device}[/cyan]")
    
    def __train_epoch(self, model, optimizer):
        model.train()
        optimizer.zero_grad()
        out = model(self.data.x, self.data.edge_index)
        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def __validate_epoch(self, model):
        model.eval()
        with torch.no_grad():
            out = model(self.data.x, self.data.edge_index)
        val_loss = F.nll_loss(out[self.data.val_mask], self.data.y[self.data.val_mask]).item()
        pred = out.argmax(dim=1)
        embeddings = out.cpu().detach()
        accs = []
        for mask in [self.data.train_mask, self.data.val_mask]:
            correct = pred[mask].eq(self.data.y[mask]).sum().item()
            accs.append(correct / mask.sum().item())
        return val_loss, accs, embeddings
    
    def get_full_training_results(self):
        return self._train_loss_over_time, self._train_acc_over_time, self._val_loss_over_time, self._val_acc_over_time
    
    def train(self):
        print("Starting Training")
        logger.info("Starting Training")
        
        # Create a Profile object
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            for epoch in range(1, self._epochs + 1):
                train_loss = self.__train_epoch(self.model, self.optimizer)
                log_system_metrics(epoch)
                val_loss, acc, embeddings = self.__validate_epoch(self.model)
                train_acc, val_acc = acc
                self._train_loss_over_time.append(train_loss)
                self._val_loss_over_time.append(val_loss)
                self._train_acc_over_time.append(train_acc)
                self._val_acc_over_time.append(val_acc)
                self._csv_writer.writerow([train_loss, train_acc, val_loss, val_acc])
                if epoch % self._print_stats_freq == 0:
                    print(f'[yellow]Epoch: {epoch:03d}, '
                        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}[/yellow]')
                    logger.info(f'Epoch: {epoch:03d}, '
                        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                if epoch % self._model_save_freq == 0:
                    print(f'Saving model at [green]epoch {epoch}[/green]')
                    logger.info(f'Saving model at epoch {epoch}')
                    torch.save(self.model, f'./model_{epoch}.pth')
            print("[bold green]Training Completed Successfully![/bold green]")
            logger.info("Training Completed Successfully!")
        
        except KeyboardInterrupt:
            print("Keyboard Interrupt detected. Saving model...")
            logger.info("Keyboard Interrupt detected. Saving model...")
            torch.save(self.model, './model_interrupted.pth')
            print("[bold yellow]Model saved as model_interrupted.pth[/bold yellow]")
            logger.info("Model saved as model_interrupted.pth")
        
        finally:
            # Disable profiler and save results
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.dump_stats('training_profile.prof')
            print("[bold green]Profiling results saved to training_profile.prof[/bold green]")
            logger.info("Profiling results saved to training_profile.prof")
            
            # Print top 20 time-consuming functions
            print("\n[bold cyan]Top 20 Time-Consuming Functions:[/bold cyan]")
            stats.strip_dirs().sort_stats('cumulative').print_stats(20)
            
            self._file.close()

@hydra.main(version_base=None, config_path="confs", config_name="training_conf")
def main(cfg):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    trainer_obj = Trainer(cfg)
    trainer_obj.train()

if __name__ == "__main__":
    main()