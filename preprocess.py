import os
import pickle
import numpy as np
import random
import torch
from path import Path
from argparse import ArgumentParser, Namespace
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from models.dataset import MNISTDataset, CIFARDataset
from collections import Counter
from torch.utils.data import Dataset
from fedlab.utils.dataset.functional import noniid_slicing, random_slicing
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

DATASET = {
    "mnist": (MNIST, MNISTDataset),
    "cifar": (CIFAR10, CIFARDataset),
}


MEAN = {
    "mnist": (0.1307,),
    "cifar": (0.4914, 0.4822, 0.4465),
}

STD = {
    "mnist": (0.3015,),
    "cifar": (0.2023, 0.1994, 0.2010),
}


def preprocess(args: Namespace) -> None:
    datasetDir = "data/" + args.dataset
    picklesDir = "data/" + args.dataset + "/pickles"

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    numOfTrainClients = int(args.numOfTrainClients)
    numOfTestClients = int(args.numOfTestClients)
    transform = transforms.Compose(
        [
            transforms.Normalize(MEAN[args.dataset], STD[args.dataset]),
        ]
    )
    targetTransform = None
    if not os.path.isdir("data"):
        os.mkdir("data")
    if not os.path.isdir(datasetDir):
        os.mkdir(datasetDir)
    if os.path.isdir(picklesDir):
        os.system(f"rm -rf {picklesDir}")
    os.mkdir(f"{picklesDir}")

    originalDataset, targetDataset = DATASET[args.dataset]
    trainset = originalDataset(
        datasetDir, train=True, download=True, transform=transforms.ToTensor()
    )
    testset = originalDataset(datasetDir, train=False, transform=transforms.ToTensor())

    num_classes = args.numOfClasses

    trainsets = randomly_alloc_classes(
        oriDataset=trainset,
        targetDataset=targetDataset,
        numOfClients=numOfTrainClients,
        transform=transform,
        target_transform=targetTransform,
    )
    testsets = randomly_alloc_classes(
        oriDataset=testset,
        targetDataset=targetDataset,
        numOfClients=numOfTestClients,
        transform=transform,
        target_transform=targetTransform,
    )

    allDatasets = trainsets + testsets
    for clientID, dataset in enumerate(allDatasets):
        with open(picklesDir + "/" + str(clientID) + ".pkl", "wb") as f:
            pickle.dump(dataset, f)
    with open(picklesDir + "/" + "seperation.pkl", "wb") as f:
        # 在这个地方划分测试集和训练集
        pickle.dump(
            {
                "train": [i for i in range(numOfTrainClients)],
                "test": [
                    i
                    for i in range(
                        numOfTrainClients, numOfTrainClients + numOfTestClients
                    )
                ],
                "total": numOfTestClients + numOfTrainClients,
            },
            f,
        )


def randomly_alloc_classes(
    oriDataset: Dataset,
    targetDataset: Dataset,
    numOfClients: int,
    transform=None,
    target_transform=None,
) -> list:
    if numOfClients == 0:
        return []
    dictUsers = random_slicing(oriDataset, numOfClients)
    datasets = []
    stats = {}
    for clientIndex, indices in dictUsers.items():
        datapoints4Client = []
        stats_tmp = []
        for i in indices:
            datapoints4Client.append((oriDataset[i][0], oriDataset[i][1]))
            stats_tmp.append(int(oriDataset[i][1]))
        datasets.append(
            targetDataset(
                datapoints4Client,
                transform=transform,
                target_transform=target_transform,
            )
        )
        stats[f"client {clientIndex}"] = dict(Counter(stats_tmp))
    return datasets


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar"],
        default="mnist",
    )
    parser.add_argument("--numOfClasses", type=int, default=10)
    parser.add_argument("--numOfTrainClients", type=int, default=10)
    parser.add_argument("--numOfTestClients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(args)
    preprocess(args)
