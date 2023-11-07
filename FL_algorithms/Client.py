import rich
import torch
import utils.basic_utils as basic_utils
from copy import deepcopy
from typing import Tuple, Union
from collections import OrderedDict
from utils.data_utils import get_dataloader
from fedlab.utils.serialization import SerializationTool


class Client:
    def __init__(
        self,
        client_id: int,
        global_model: torch.nn.Module,
        criterion: Union[torch.nn.CrossEntropyLoss, torch.nn.MSELoss],
        batch_size: int,
        dataset: str,
        valset_ratio: float,
        logger: rich.console.Console,
        gpu: int,
        eta: float,
        stats: list,
    ):
        if gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.logger = logger
        self.criterion = criterion
        self.id = client_id
        self.model = deepcopy(global_model)
        self.trainloader, self.valloader = get_dataloader(
            dataset, client_id, batch_size, valset_ratio
        )
        self.iter_trainloader = iter(self.trainloader)
        self.eta = eta
        self.stats = stats

    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)
        return x.to(self.device), y.to(self.device)

    def compute_grad(
        self,
        model: torch.nn.Module,
        data_batch: Tuple[torch.Tensor, torch.Tensor],
        v: Union[Tuple[torch.Tensor, ...], None] = None,
        second_order_grads=False,
    ):
        x, y = data_batch
        if second_order_grads:
            frz_model_params = deepcopy(model.state_dict())
            delta = 1e-3
            dummy_model_params_1 = OrderedDict()
            dummy_model_params_2 = OrderedDict()
            with torch.no_grad():
                for (layer_name, param), grad in zip(model.named_parameters(), v):
                    dummy_model_params_1.update({layer_name: param + delta * grad})
                    dummy_model_params_2.update({layer_name: param - delta * grad})

            model.load_state_dict(dummy_model_params_1, strict=False)
            logit_1 = model(x)
            loss_1 = self.criterion(logit_1, y)
            grads_1 = torch.autograd.grad(loss_1, model.parameters())

            model.load_state_dict(dummy_model_params_2, strict=False)
            logit_2 = model(x)
            loss_2 = self.criterion(logit_2, y)
            grads_2 = torch.autograd.grad(loss_2, model.parameters())

            model.load_state_dict(frz_model_params)

            grads = []
            with torch.no_grad():
                for g1, g2 in zip(grads_1, grads_2):
                    grads.append((g1 - g2) / (2 * delta))
            return grads

        else:
            logit = model(x)
            loss = self.criterion(logit, y)
            grads = torch.autograd.grad(loss, model.parameters())
            return grads, loss

    def train(self):
        return SerializationTool.serialize_model(self.model)

    def log(self, loss, acc, if_log, training_round=-1, training_type="global"):
        if if_log == True:
            self.logger.log(
                "client [{}] [red]loss: {:.4f}   [blue]acc: {:.2f}%".format(
                    self.id,
                    loss,
                    acc * 100.0,
                )
            )
        if training_round != -1:
            self.stats.append(
                [
                    "train",
                    training_type,
                    training_round,
                    self.id,
                    float(loss),
                    float(acc),
                ]
            )
        else:
            self.stats.append(
                [
                    "test",
                    "",
                    training_round,
                    self.id,
                    float(loss),
                    float(acc),
                ]
            )


class FedAvgClient(Client):
    def __init__(
        self,
        client_id: int,
        eta: float,
        global_model: torch.nn.Module,
        criterion: Union[torch.nn.CrossEntropyLoss, torch.nn.MSELoss],
        batch_size: int,
        dataset: str,
        local_epochs: int,
        valset_ratio: float,
        logger: rich.console.Console,
        gpu: int,
        stats: list,
    ):
        super(FedAvgClient, self).__init__(
            client_id,
            global_model,
            criterion,
            batch_size,
            dataset,
            valset_ratio,
            logger,
            gpu,
            eta,
            stats,
        )
        self.local_epochs = local_epochs
        self.criterion = criterion

    def train(
        self,
        global_model: torch.nn.Module,
        if_log: bool,
        eval_while_training=False,
        training_round=-1,
    ):
        self.model.load_state_dict(global_model.state_dict())
        self._train()
        if eval_while_training:
            loss, acc, pre_y_list = basic_utils.eval(
                self.model, self.valloader, self.criterion, self.device
            )
        self.log(loss, acc, if_log, training_round)
        return SerializationTool.serialize_model(self.model)

    def _train(self):
        for _ in range(self.local_epochs):
            data_batch = self.get_data_batch()
            grads, loss = self.compute_grad(self.model, data_batch)
            for param, grad in zip(self.model.parameters(), grads):
                param.data.sub_(self.eta * grad)

    def eval(self, global_model: torch.nn.Module, if_log: bool):
        self.model.load_state_dict(global_model.state_dict())
        loss, acc, pre_y_list = basic_utils.eval(
            self.model, self.valloader, self.criterion, self.device
        )
        self.log(loss, acc, if_log)
        return {"loss": loss, "acc": acc, "pred_y_list": pre_y_list}
