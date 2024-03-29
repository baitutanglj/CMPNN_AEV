from argparse import Namespace
import random
from typing import Callable, Dict, Iterator, List, Optional, Union

import numpy as np
from torch.utils.data.dataset import Dataset
from rdkit import Chem

from .scaler import StandardScaler
from chemprop.features import get_features_generator


class MoleculeDatapoint:
    """A MoleculeDatapoint contains a single molecule and its associated features and targets."""

    def __init__(self,
                 smiles: str,
                 targets: List[Optional[float]] = None,
                 args: Namespace = None,
                 features: np.ndarray = None,
                 atom_features: np.ndarray = None,
                 atom_descriptors: np.ndarray = None,
                 bond_features: np.ndarray = None,
                 overwrite_default_atom_features: bool = False,
                 overwrite_default_bond_features: bool = False):
        """
        Initializes a MoleculeDatapoint, which contains a single molecule.

        :param line: A list of strings generated by separating a line in a data CSV file by comma.
        :param args: Arguments.
        :param features: A numpy array containing additional features (ex. Morgan fingerprint).
        :param use_compound_names: Whether the data CSV includes the compound name on each line.
        :param atom_descriptors: A numpy array containing additional atom descriptors to featurize the molecule
        :param bond_features: A numpy array containing additional bond features to featurize the molecule
        :param overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features
        :param overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features
        """
        if args is not None:
            self.features_generator = args.features_generator
            self.args = args
        else:
            self.features_generator = self.args = None

        if features is not None and self.features_generator is not None:
            raise ValueError('Currently cannot provide both loaded features and a features generator.')

        self.features = features

        self.smiles = smiles  # str
        # Create targets
        self.targets = [float(x) if x != '' else None for x in targets]
        # self.mol = Chem.MolFromSmiles(self.smiles)
        ##################my addition#######################
        if args.ligand_file_type == 'smiles':
            self.mol = Chem.MolFromSmiles(self.smiles)
        elif args.ligand_file_type == 'pdb':
            self.mol = Chem.MolFromPDBFile(args.datapaths+'/'+self.smiles)
        elif args.ligand_file_type == 'sdf':
            self.mol = Chem.SDMolSupplier(args.datapaths+'/'+self.smiles)[0]
        elif args.ligand_file_type == 'mol2':
            self.mol = Chem.MolFromMol2File(args.datapaths+'/'+self.smiles)
        self.atom_descriptors = atom_descriptors
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features
        ####################################################

        # Generate additional features if given a generator
        if self.features_generator is not None:
            self.features = []

            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                if self.mol is not None and self.mol.GetNumHeavyAtoms() > 0:
                    self.features.extend(features_generator(self.mol))

            self.features = np.array(self.features)

        # Fix nans in features
        replace_token = 0
        if self.features is not None:
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        # Fix nans in atom_descriptors
        if self.atom_descriptors is not None:
            self.atom_descriptors = np.where(np.isnan(self.atom_descriptors), replace_token, self.atom_descriptors)

        # Fix nans in atom_features
        if self.atom_features is not None:
            self.atom_features = np.where(np.isnan(self.atom_features), replace_token, self.atom_features)

        # Fix nans in bond_descriptors
        if self.bond_features is not None:
            self.bond_features = np.where(np.isnan(self.bond_features), replace_token, self.bond_features)

        # Save a copy of the raw features and targets to enable different scaling later on
        self.raw_features, self.raw_targets = self.features, self.targets
        self.raw_atom_descriptors, self.raw_atom_features, self.raw_bond_features = \
            self.atom_descriptors, self.atom_features, self.bond_features

    def set_features(self, features: np.ndarray):
        """
        Sets the features of the molecule.

        :param features: A 1-D numpy array of features for the molecule.
        """
        self.features = features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[float]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets

    ##################my addition#########################
    def set_atom_descriptors(self, atom_descriptors: np.ndarray) -> None:
        """
        Sets the atom descriptors of the molecule.

        :param atom_descriptors: A 1D numpy array of features for the molecule.
        """
        self.atom_descriptors = atom_descriptors

    def set_atom_features(self, atom_features: np.ndarray) -> None:
        """
        Sets the atom features of the molecule.

        :param atom_features: A 1D numpy array of features for the molecule.
        """
        self.atom_features = atom_features

    def set_bond_features(self, bond_features: np.ndarray) -> None:
        """
        Sets the bond features of the molecule.

        :param bond_features: A 1D numpy array of features for the molecule.
        """
        self.bond_features = bond_features

    def extend_features(self, features: np.ndarray) -> None:
        """
        Extends the features of the molecule.

        :param features: A 1D numpy array of extra features for the molecule.
        """
        self.features = np.append(self.features, features) if self.features is not None else features

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        self.features, self.targets = self.raw_features, self.raw_targets
        self.atom_descriptors, self.atom_features, self.bond_features = \
            self.raw_atom_descriptors, self.raw_atom_features, self.raw_bond_features
    ######################################################

class MoleculeDataset(Dataset):
    """A MoleculeDataset contains a list of molecules and their associated features and targets."""

    def __init__(self, data: List[MoleculeDatapoint]):
        """
        Initializes a MoleculeDataset, which contains a list of MoleculeDatapoints (i.e. a list of molecules).

        :param data: A list of MoleculeDatapoints.
        """
        self.data = data
        self.args = self.data[0].args if len(self.data) > 0 else None
        self.scaler = None
        self.atom_scaler = None
        self.bond_scaler = None

    def compound_names(self) -> List[str]:
        """
        Returns the compound names associated with the molecule (if they exist).

        :return: A list of compound names or None if the dataset does not contain compound names.
        """
        if len(self.data) == 0 or self.data[0].compound_name is None:
            return None

        return [d.compound_name for d in self.data]

    def smiles(self) -> List[str]:
        """
        Returns the smiles strings associated with the molecules.

        :return: A list of smiles strings.
        """
        return [d.smiles for d in self.data]
    
    def mols(self) -> List[Chem.Mol]:
        """
        Returns the RDKit molecules associated with the molecules.

        :return: A list of RDKit Mols.
        """
        return [d.mol for d in self.data]

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        return [d.features for d in self.data]

    def targets(self) -> List[List[float]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats containing the targets.
        """
        return [d.targets for d in self.data]

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self.data[0].num_tasks() if len(self.data) > 0 else None

    def features_size(self) -> int:
        """
        Returns the size of the features array associated with each molecule.

        :return: The size of the features.
        """
        return len(self.data[0].features) if len(self.data) > 0 and self.data[0].features is not None else None

    ################my addition###################
    def atom_features(self) -> List[np.ndarray]:
        """
        Returns the atom descriptors associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the atom descriptors
                 for each molecule or None if there are no features.
        """
        if len(self.data) == 0 or self.data[0].atom_features is None:
            return None

        return [d.atom_features for d in self.data]

    def atom_descriptors(self) -> List[np.ndarray]:
        """
        Returns the atom descriptors associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the atom descriptors
                 for each molecule or None if there are no features.
        """
        if len(self.data) == 0 or self.data[0].atom_descriptors is None:
            return None

        return [d.atom_descriptors for d in self.data]

    def bond_features(self) -> List[np.ndarray]:
        """
        Returns the bond features associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the bond features
                 for each molecule or None if there are no features.
        """
        if len(self.data) == 0 or self.data[0].bond_features is None:
            return None

        return [d.bond_features for d in self.data]

    def atom_descriptors_size(self) -> int:
        """
        Returns the size of custom additional atom descriptors vector associated with the molecules.

        :return: The size of the additional atom descriptor vector.
        """
        return len(self.data[0].atom_descriptors[0]) \
            if len(self.data) > 0 and self.data[0].atom_descriptors is not None else None

    def atom_features_size(self) -> int:
        """
        Returns the size of custom additional atom features vector associated with the molecules.

        :return: The size of the additional atom feature vector.
        """
        return len(self.data[0].atom_features[0]) \
            if len(self.data) > 0 and self.data[0].atom_features is not None else None

    def bond_features_size(self) -> int:
        """
        Returns the size of custom additional bond features vector associated with the molecules.

        :return: The size of the additional bond feature vector.
        """
        return len(self.data[0].bond_features[0]) \
            if len(self.data) > 0 and self.data[0].bond_features is not None else None
    ################################

    def shuffle(self, seed: int = None):
        """
        Shuffles the dataset.

        :param seed: Optional random seed.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
    
    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        """
        Normalizes the features of the dataset using a StandardScaler (subtract mean, divide by standard deviation).

        If a scaler is provided, uses that scaler to perform the normalization. Otherwise fits a scaler to the
        features in the dataset and then performs the normalization.

        :param scaler: A fitted StandardScaler. Used if provided. Otherwise a StandardScaler is fit on
        this dataset and is then used.
        :param replace_nan_token: What to replace nans with.
        :return: A fitted StandardScaler. If a scaler is provided, this is the same scaler. Otherwise, this is
        a scaler fit on this dataset.
        """
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        if scaler is not None:
            self.scaler = scaler

        elif self.scaler is None:
            features = np.vstack([d.features for d in self.data])
            self.scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self.scaler.fit(features)

        for d in self.data:
            d.set_features(self.scaler.transform(d.features.reshape(1, -1))[0])

        return self.scaler
    ##############my addition#################
    def atom_normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        """
        Normalizes the features of the dataset using a StandardScaler (subtract mean, divide by standard deviation).

        If a scaler is provided, uses that scaler to perform the normalization. Otherwise fits a scaler to the
        features in the dataset and then performs the normalization.

        :param scaler: A fitted StandardScaler. Used if provided. Otherwise a StandardScaler is fit on
        this dataset and is then used.
        :param replace_nan_token: What to replace nans with.
        :return: A fitted StandardScaler. If a scaler is provided, this is the same scaler. Otherwise, this is
        a scaler fit on this dataset.
        """
        if len(self.data) == 0 or \
                (self.data[0].atom_features is None and self.data[0].atom_descriptors is None):
            return None

        if scaler is not None:
            self.atom_scaler = scaler

        elif self.atom_scaler is None:
            if not self.data[0].atom_descriptors is None:
                features = np.vstack([d.raw_atom_descriptors for d in self.data])
            elif not self.data[0].atom_features is None:
                features = np.vstack([d.raw_atom_features for d in self.data])
            self.atom_scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self.atom_scaler.fit(features)

        if not self.data[0].atom_descriptors is None:
            for d in self.data:
                d.set_atom_descriptors(self.atom_scaler.transform(d.raw_atom_descriptors))
        elif not self.data[0].atom_features is None:
            for d in self.data:
                d.set_atom_features(self.atom_scaler.transform(d.raw_atom_features))

        return self.atom_scaler

    def bond_normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        """
        Normalizes the features of the dataset using a StandardScaler (subtract mean, divide by standard deviation).

        If a scaler is provided, uses that scaler to perform the normalization. Otherwise fits a scaler to the
        features in the dataset and then performs the normalization.

        :param scaler: A fitted StandardScaler. Used if provided. Otherwise a StandardScaler is fit on
        this dataset and is then used.
        :param replace_nan_token: What to replace nans with.
        :return: A fitted StandardScaler. If a scaler is provided, this is the same scaler. Otherwise, this is
        a scaler fit on this dataset.
        """
        if len(self.data) == 0 or self.data[0].bond_features is None:
            return None

        if scaler is not None:
            self.bond_scaler = scaler

        elif self.bond_scaler is None:
            features = np.vstack([d.features for d in self.data])
            self.bond_scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self.bond_scaler.fit(features)

        for d in self.data:
            d.set_features(self.bond_scaler.transform(d.features.reshape(1, -1))[0])

        return self.bond_scaler
    ##########################################
    ##############my addition#################
    # def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0,
    #                        scale_atom_descriptors: bool = False, scale_bond_features: bool = False) -> StandardScaler:
    #     """
    #     Normalizes the features of the dataset using a :class:`~chemprop.data.StandardScaler`.
    #
    #     The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
    #     for each feature independently.
    #
    #     If a :class:`~chemprop.data.StandardScaler` is provided, it is used to perform the normalization.
    #     Otherwise, a :class:`~chemprop.data.StandardScaler` is first fit to the features in this dataset
    #     and is then used to perform the normalization.
    #
    #     :param scaler: A fitted :class:`~chemprop.data.StandardScaler`. If it is provided it is used,
    #                    otherwise a new :class:`~chemprop.data.StandardScaler` is first fitted to this
    #                    data and is then used.
    #     :param replace_nan_token: A token to use to replace NaN entries in the features.
    #     :param scale_atom_descriptors: If the features that need to be scaled are atom features rather than molecule.
    #     :param scale_bond_features: If the features that need to be scaled are bond descriptors rather than molecule.
    #     :return: A fitted :class:`~chemprop.data.StandardScaler`. If a :class:`~chemprop.data.StandardScaler`
    #              is provided as a parameter, this is the same :class:`~chemprop.data.StandardScaler`. Otherwise,
    #              this is a new :class:`~chemprop.data.StandardScaler` that has been fit on this dataset.
    #     """
    #     if len(self.data) == 0 or \
    #             (self.data[0].features is None and not scale_bond_features and not scale_atom_descriptors):
    #         return None
    #
    #     if scaler is not None:
    #         self.scaler = scaler
    #
    #     elif self.scaler is None:
    #         if scale_atom_descriptors and not self.data[0].atom_descriptors is None:
    #             features = np.vstack([d.raw_atom_descriptors for d in self.data])
    #         elif scale_atom_descriptors and not self.data[0].atom_features is None:
    #             features = np.vstack([d.raw_atom_features for d in self.data])
    #         elif scale_bond_features:
    #             features = np.vstack([d.raw_bond_features for d in self.data])
    #         else:
    #             features = np.vstack([d.raw_features for d in self.data])
    #         self.scaler = StandardScaler(replace_nan_token=replace_nan_token)
    #         self.scaler.fit(features)
    #
    #     if scale_atom_descriptors and not self.data[0].atom_descriptors is None:
    #         for d in self.data:
    #             d.set_atom_descriptors(self.scaler.transform(d.raw_atom_descriptors))
    #     elif scale_atom_descriptors and not self.data[0].atom_features is None:
    #         for d in self.data:
    #             d.set_atom_features(self.scaler.transform(d.raw_atom_features))
    #     elif scale_bond_features:
    #         for d in self.data:
    #             d.set_bond_features(self.scaler.transform(d.raw_bond_features))
    #     else:
    #         for d in self.data:
    #             d.set_features(self.scaler.transform(d.raw_features.reshape(1, -1))[0])
    #
    #     return self.scaler
    ##########################################


    def set_targets(self, targets: List[List[float]]):
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats containing targets for each molecule. This must be the
        same length as the underlying dataset.
        """
        assert len(self.data) == len(targets)
        for i in range(len(self.data)):
            self.data[i].set_targets(targets[i])

    def sort(self, key: Callable):
        """
        Sorts the dataset using the provided key.

        :param key: A function on a MoleculeDatapoint to determine the sorting order.
        """
        self.data.sort(key=key)

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e. the number of molecules).

        :return: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        """
        Gets one or more MoleculeDatapoints via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A MoleculeDatapoint if an int is provided or a list of MoleculeDatapoints if a slice is provided.
        """
        return self.data[item]
