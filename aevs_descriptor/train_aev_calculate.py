import os
from pathlib import Path
import json
import pandas as pd
import torch
import torchani
from ael import loaders
import argparsers
from ael import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_aev_from_loader(args):
    result_dir = os.path.dirname(args.outpath)
    if Path(args.outpath).exists():
        os.remove(args.outpath)
    else:
        Path(result_dir).mkdir(parents=True, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.chemap is not None:
        cmap = json.loads(args.chemap)
    else:
        cmap = None

    # Distance 0.0 produces a segmentation fault (see MDAnalysis#2656)
    data = loaders.PDBData(
        args.trainfile,
        args.distance,
        args.datapaths,
        cmap,
        desc="dataset",
        removeHs=args.removeHs,
        no_contain_header = args.no_contain_header
    )

    if cmap is not None:
        path = os.path.join(result_dir, "cmap.json")
        with open(path, "w") as fout:
            json.dump(cmap, fout)

    # Compute map of atomic numbers to indices from species
    amap = loaders.anummap(data.species)
    # Save amap to JSON file
    utils.save_amap(amap, path=os.path.join(result_dir, "amap.json"))

    # Transform atomic number to species in data
    data.atomicnums_to_idxs(amap)

    n_species = len(amap)

    # Radial coefficients
    EtaR = torch.tensor([args.EtaR], device=device)
    RsR = torch.tensor(args.RsR, device=device)
    # Angular coefficients
    RsA = torch.tensor(args.RsA, device=device)
    EtaA = torch.tensor([args.EtaA], device=device)
    TsA = torch.tensor(args.TsA, device=device)
    Zeta = torch.tensor([args.Zeta], device=device)

    # Define AEVComputer
    AEVC = torchani.AEVComputer(args.RcR, args.RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, n_species)
    # Save AEVComputer
    utils.saveAEVC(AEVC, n_species, path=os.path.join(result_dir, "aevc.pth"))

    aevs = []
    aevs_species = []
    for maskLIG, species, coordinates in zip(data.maskLIG, data.species, data.coordinates):
        # Move everything to device
        species = torch.unsqueeze(species, 0).to(device)
        coordinates = torch.unsqueeze(coordinates, 0).to(device)
        aev = AEVC.forward((species, coordinates)).aevs
        aevs.append(aev[0,maskLIG].cpu().numpy())
        aevs_species.append(species[0,maskLIG].cpu().numpy())
    df = pd.DataFrame({'features':aevs, 'species':aevs_species},index=data.ids)
    # df.to_pickle('/mnt/home/linjie/projects/aescore/aevs_descriptor/result/aevs.pkl')
    df.to_pickle(args.outpath)
    return aevs


if __name__ == "__main__":
    args = argparsers.trainparser(default="BP")
    aevs = train_aev_from_loader(args)
    print('ok')
    # pkldata = pd.read_pickle(args.outpath)