import sys

sys.path.append("FL_algorithms")
import rich
import torch
import utils.basic_utils as basic_utils
from copy import deepcopy
from typing import Tuple, Union
from collections import OrderedDict
from utils.data_utils import get_dataloader
from fedlab.utils.serialization import SerializationTool
from Client import Client


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
