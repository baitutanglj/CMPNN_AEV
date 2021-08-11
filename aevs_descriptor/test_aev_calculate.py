import os
import shutil
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torchani
from typing import List, Optional, Tuple, Union
from ael import loaders
import argparsers
from ael import utils
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_aev_from_loader(args):
    if Path(args.outpath).exists():
        os.remove(args.outpath)
    else:
        pass

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.chemap is not None:
        with open(args.chemap, "r") as fin:
            cmap = json.load(fin)
    else:
        cmap = None

    # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
    data: Union[loaders.PDBData, loaders.VSData] = loaders.PDBData(
        args.testfile,
        args.distance,
        args.datapaths,
        cmap,
        desc="dataset",
        removeHs=args.removeHs,
        no_contain_header=args.no_contain_header
    )

    # load  map of atomic numbers to indices from species
    amap = utils.load_amap(args.amap)
    # Transform atomic number to species in data
    data.atomicnums_to_idxs(amap)

    n_species = len(amap)

    # load AEVComputer
    AEVC = utils.loadAEVC(args.aev)

    aevs = []
    for maskLIG, species, coordinates in zip(data.maskLIG, data.species, data.coordinates):
        # Move everything to device
        species = torch.unsqueeze(species, 0).to(device)
        coordinates = torch.unsqueeze(coordinates, 0).to(device)
        aev = AEVC.forward((species, coordinates)).aevs
        aevs.append(aev[0,maskLIG].cpu().numpy())
    df = pd.DataFrame(aevs,index=data.ids)
    # df.to_pickle('/mnt/home/linjie/projects/aescore/aevs_descriptor/result/aevs.pkl')
    df.to_pickle(args.outpath)
    return aevs


if __name__ == "__main__":
    args = argparsers.predictparser()
    aevs = test_aev_from_loader(args)
    print('ok')
    # pkldata = pd.read_pickle(os.path.join(args.outpath,'aevs.pkl'))