import os
import sys
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_mytest)
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
from rdkit import Chem
from chemprop.features.featurization import ATOM_FEATURES, atom_features


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2

atom_type= ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
              'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
              'LEU', 'LYS', 'MET', 'PHE', 'PRO',
              'SER', 'THR', 'TRP', 'TYR', 'VAL']

def onek_encoding_atom_type(residue):
    encoding = [0] * (len(atom_type))
    idx = atom_type.index(residue)
    encoding[idx] = 1
    return encoding


def generate_atomic_types(args, pdb_code, resselection, aev_protein, maskLIG):
    new_aev_protein = np.zeros_like(aev_protein)
    if args.add_atom_descriptors:
        mol = Chem.MolFromPDBFile(os.path.join(args.datapaths[0], pdb_code, pdb_code + '_protein-nhm.pdb'),
                                  sanitize=False, removeHs=False)
        atom_descriptors_matrix = np.zeros((aev_protein.shape[0],ATOM_FDIM))
        new_aev_protein = np.hstack((new_aev_protein,atom_descriptors_matrix))
    if args.add_atom_type:
        new_aev_protein_matrix = np.zeros((aev_protein.shape[0],len(atom_type)))
        new_aev_protein = np.hstack((new_aev_protein, new_aev_protein_matrix))
    idx = resselection.ids[~maskLIG]
    idx = (idx - 1).tolist()
    residues = resselection.resnames[~maskLIG]

    for i, (atomidx,residue) in enumerate(zip(idx,residues)):
        descriptors_list = []
        if args.add_atom_descriptors:
            atom = mol.GetAtomWithIdx(atomidx)
            f_atoms = atom_features(atom)
            descriptors_list.extend(f_atoms)
        if args.add_atom_type:
            t_atoms = onek_encoding_atom_type(residue)
            descriptors_list.extend(t_atoms)
        new_aev_protein[i] = np.hstack((aev_protein[i],
                                        np.array(descriptors_list,dtype=aev_protein.dtype)))
        # new_aev_protein[i] = torch.cat((aev_protein[i], torch.tensor(f_atoms,dtype=aev_protein.dtype)))

    return new_aev_protein


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
    aevs_species = []
    aevs_protein = []
    aevs_species_protein = []

    ##aev with ligand and protein
    for i, (maskLIG, species, coordinates) in enumerate(zip(data.maskLIG, data.species, data.coordinates)):
        # Move everything to device
        species = torch.unsqueeze(species, 0).to(device)
        coordinates = torch.unsqueeze(coordinates, 0).to(device)
        aev = AEVC.forward((species, coordinates)).aevs
        aevs.append(aev[0,maskLIG].cpu().numpy())
        aevs_species.append(species[0,maskLIG].cpu().numpy())

        ####protein###
        aev_protein = aev[0,~maskLIG].cpu().numpy()
        if args.add_atom_descriptors or args.add_atom_type:
            aev_protein = generate_atomic_types(args, data.ids[i], data.resselection[i], aev_protein, maskLIG)
        aevs_protein.append(aev_protein)
        aevs_species_protein.append(species[0,~maskLIG].cpu().numpy())

    df = pd.DataFrame({'features':aevs, 'species':aevs_species},index=data.ids)
    df_protein = pd.DataFrame({'features': aevs_protein,'species':aevs_species}, index=data.ids)
    # df.to_pickle('/mnt/home/linjie/projects/aescore/aevs_result/result/aevs.pkl')
    df.to_pickle(args.outpath)
    df_protein.to_pickle(os.path.join(os.path.dirname(args.outpath), 'test_aevs_protein.pkl'))
    return aevs

if __name__ == "__main__":
    args = argparsers.predictparser()
    aevs = test_aev_from_loader(args)
    print('ok')