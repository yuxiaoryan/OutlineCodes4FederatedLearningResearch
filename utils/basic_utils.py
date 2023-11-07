import torch
import random
import numpy as np
from typing import Iterator, Tuple, Union
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    ##meta infomation of an experiment
    parser.add_argument(
        "--experiment_number",
        type=int,
        default=1,
        help="Giving a number to an experiment setting helps build kind of order among a series of experiments",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="FedAvg",
    )
    parser.add_argument(
        "--dataset", type=str, choices=["mnist", "cifar"], default="mnist"
    )
    ##public args
    parser.add_argument("--global_epochs", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument(
        "--hf",
        type=int,
        default=1,
        help="0 for performing Per-FedAvg(FO), others for Per-FedAvg(HF)",
    )
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument(
        "--valset_ratio",
        type=float,
        default=0.1,
        help="Proportion of val set in the entire client local dataset",
    )

    parser.add_argument("--client_num_per_round", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="Non-zero value for using gpu, 0 for using cpu",
    )
    parser.add_argument(
        "--eval_while_training",
        type=int,
        default=1,
        help="Non-zero value for performing local evaluation before and after local training",
    )
    parser.add_argument(
        "--log",
        type=int,
        default=1,
        help="Whether to log the process of training clients or not",
    )
    return parser.parse_args()


@torch.no_grad()
def eval(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Union[torch.nn.MSELoss, torch.nn.CrossEntropyLoss],
    device=torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_loss = 0
    num_samples = 0
    acc = 0
    pred_y_list = []
    tt = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logit = model(x)
        # total_loss += criterion(logit, y) / y.size(-1)
        total_loss += criterion(logit, y)
        pred = torch.softmax(logit, -1).argmax(-1)
        acc += torch.eq(pred, y).int().sum()
        num_samples += y.size(-1)
        pred_y_list.append(
            {"pred": [int(i) for i in list(pred)], "y": [int(i) for i in list(y)]}
        )
    return total_loss, acc / num_samples, pred_y_list


def insert_str(loc: int, s1: str, s2: str):
    a = list(s1)
    s1 = a.insert(loc, s2)
    return "".join(a)


def list_merge(l1: list, l2: list, sort_bool: int = 1):
    a = []
    for v1 in l1:
        a.append(v1)
    for v2 in l2:
        a.append(v2)
    if sort_bool != 0:
        a.sort(reverse=True if a == 1 else False)
    return a


def fix_random_seed(seed: int):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
