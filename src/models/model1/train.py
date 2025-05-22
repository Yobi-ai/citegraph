import torch
import torch.nn.functional as F
from dataloader import Dataset
from model import GCN


class Trainer:
    def __init__(self, epochs = 5000, model_save_freq = 1000, print_stats_freq = 50):
        self._epochs = epochs
        self._model_save_freq = model_save_freq
        self._print_stats_freq = print_stats_freq

        self._embeddings_over_time = []

        self._train_loss_over_time = []
        self._val_loss_over_time = []
        self._train_acc_over_time = []
        self._val_acc_over_time = []

        self.__intialize_objects()

    def __intialize_objects(self):
        dataloader = Dataset()
        self.data = dataloader.load_cora()
        self.model = GCN(self.data.num_node_features, 717, self.data.num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)

        self.__move_to_device()

    def __move_to_device(self):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self._device)
        self.data.to(self._device)
    
    def __train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index)
        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def __validate_epoch(self, model):
        self.model.eval()
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
        for epoch in range(1, self._epochs + 1):
            train_loss = self.__train_epoch(self.model)
            val_loss, acc, embeddings = self.__validate_epoch(self.model)
            #embeddings_over_time.append(embeddings)
            train_acc, val_acc = acc
            self._train_loss_over_time.append(train_loss)
            self._val_loss_over_time.append(val_loss)
            self._train_acc_over_time.append(train_acc)
            self._val_acc_over_time.append(val_acc)
            if epoch % self._print_stats_freq == 0:
                print(f'Epoch: {epoch:03d}, '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            if epoch % self._model_save_freq == 0:
                print(f'Saving model at epoch {epoch}')
                torch.save(self.model, f'./model_{epoch}.pth')