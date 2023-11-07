import sys

sys.path.append("utils")

import torch
import random
import os
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from rich.console import Console
from rich.progress import track
from utils.basic_utils import get_args, fix_random_seed
from models.mlp import get_model
from FL_algorithms.FedAvgClient import FedAvgClient
from utils.data_utils import get_client_id_indices
import pickle
import sys


def clients_generating(client_id, args, global_model, logger, stats, is_train=True):
    if args.algorithm == "FedAvg":
        return FedAvgClient(
            client_id=client_id,
            eta=args.eta,
            global_model=global_model,
            criterion=torch.nn.CrossEntropyLoss(),
            batch_size=args.batch_size,
            dataset=args.dataset,
            local_epochs=args.local_epochs,
            valset_ratio=args.valset_ratio if is_train == True else 1,
            logger=logger,
            gpu=args.gpu,
            stats=stats,
        )


if __name__ == "__main__":
    # print(sys.path)
    args = get_args()
    fix_random_seed(args.seed)
    if os.path.isdir("./log") == False:
        os.mkdir("./log")
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    global_model = get_model(args.dataset, device)
    logger = Console(record=args.log)
    logger.log(f"Arguments:", dict(args._get_kwargs()))
    stats_list = []
    (clients_4_training, clients_4_eval, client_num_in_total) = get_client_id_indices(
        args.dataset
    )
    # exit()
    # init clients
    clients = [
        clients_generating(
            client_id=client_id,
            args=args,
            global_model=global_model,
            logger=logger,
            is_train=True,
            stats=stats_list,
        )
        for client_id in clients_4_training
    ] + [
        clients_generating(
            client_id=client_id,
            args=args,
            global_model=global_model,
            logger=logger,
            is_train=False,
            stats=stats_list,
        )
        for client_id in clients_4_eval
    ]
    # training
    # logger.log("=" * 20, "TRAINING", "=" * 20, style="bold red")
    stats_columns = [
        "train_or_test",
        "training_stage",
        "training_round",
        "client_id",
        "loss",
        "acc",
    ]

    if args.algorithm in ("FedAvg"):
        for _ in track(
            range(args.global_epochs),
            "Training...",
            console=logger,
            disable=False,
        ):
            # select clients
            selected_clients = random.sample(
                clients_4_training, args.client_num_per_round
            )
            model_params_cache = []
            # client local training
            for client_id in selected_clients:
                serialized_model_params = clients[client_id].train(
                    global_model=global_model,
                    if_log=args.log == 1,
                    eval_while_training=args.eval_while_training,
                    training_round=_,
                )
                model_params_cache.append(serialized_model_params)

            # aggregate model parameters
            aggregated_model_params = Aggregators.fedavg_aggregate(model_params_cache)
            SerializationTool.deserialize_model(global_model, aggregated_model_params)
            if args.log == 1:
                logger.log("=" * 60)
    # start testing
    pred_y_list_dict = {}
    weights_4_global_training = []
    loss_after = {}
    acc_after = {}
    for client_id in track(
        clients_4_eval, "Evaluating...", console=logger, disable=False
    ):
        stats = clients[client_id].eval(
            global_model=clients[client_id - len(clients_4_training)].model
            if args.algorithm == "FedSelf"
            else global_model,
            if_log=args.log == 1,
        )
        loss_after[client_id] = stats["loss"]
        acc_after[client_id] = stats["acc"]
        pred_y_list_dict[client_id] = stats["pred_y_list"]
    logger.log("=" * 20, "RESULTS", "=" * 20, style="bold green")
    logger.log(f"loss: {(sum(loss_after.values()) / len(loss_after)):.4f}")
    logger.log(f"acc: {(sum(acc_after.values()) * 100.0 / len(acc_after)):.2f}%")
    if os.path.isdir("./log/ex{}".format(args.experiment_number)) == False:
        os.mkdir("./log/ex{}".format(args.experiment_number))
    storage_path = "log/ex{}/{}_{}_stats.pkl".format(
        args.experiment_number, args.algorithm, args.dataset
    )
    with open(
        storage_path,
        "wb",
    ) as f:
        # 在这个地方划分测试集和训练集
        pickle.dump(
            {
                "stats_columns": stats_columns,
                "stats_list": stats_list,
                "pred_y_list_dict": pred_y_list_dict,
                "weights_4_global_training": weights_4_global_training,
            },
            f,
        )
