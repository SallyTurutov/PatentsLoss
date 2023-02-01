import rdkit
from rdkit import Chem, DataStructs
import rdkit.Chem.QED as QED
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import pickle
import random

# my files
from dataset.DRD2.DRD2_predictor import DRD2


# disable rdkit error messages
def rdkit_no_error_print():
    rdkit.rdBase.DisableLog('rdApp.*')


# returns None if molecule_SMILES string does not represent legal molecule
def smiles2mol(molecule_SMILES):
    mol = Chem.MolFromSmiles(molecule_SMILES)
    if mol is None:
        raise Exception()
    return mol


# raise exception if molecule_SMILES string does not represent legal molecule,
# otherwise return its fingerprints
def smiles2fingerprint(molecule_SMILES, radius=2, nBits=2048, useChirality=False, fp_translator=False):
    try:
        mol = smiles2mol(molecule_SMILES)
        if fp_translator is True:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useChirality=useChirality)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useChirality=useChirality)
        return fp
    except Exception as e:
        raise Exception()


# initialize property calculation
def property_init(args):
    print("Start Initialization")
    if args.property is 'DRD2':
        global DRD2_pred
        DRD2_pred = DRD2()

    global density_500000_dict
    global patentsFp_500000_list

    density_500000_path = 'dataset/PL/densityProbs500000.pkl'
    density_500000_file = open(density_500000_path, "rb")
    density_500000_dict = pickle.load(density_500000_file)
    density_500000_file.close()

    patentsFp_500000_path = 'dataset/PL/patentsFp500000.pkl'
    patentsFp_500000_file = open(patentsFp_500000_path, "rb")
    patentsFp_500000_dict = pickle.load(patentsFp_500000_file)
    patentsFp_500000_list = list(patentsFp_500000_dict.items())
    patentsFp_500000_file.close()

    print("Done Initialization")


# raise exception if molecule_SMILES string does not represent legal molecule
def property_calc(molecule_SMILES, property):
    mol = smiles2mol(molecule_SMILES)
    if property is 'QED':
        return QED.qed(mol)
    if property is 'DRD2':
        return DRD2_pred.get_score(mol)
    if property is 'PL':
        return calc_PL_score(molecule_SMILES)


# raise exception if at least one of the molecule_SMILES strings does not represent legal molecule
def similarity_calc(molecule_SMILES_1, molecule_SMILES_2, noException=False):
    try:
        fp_mol_1 = smiles2fingerprint(molecule_SMILES_1)
        fp_mol_2 = smiles2fingerprint(molecule_SMILES_2)
        return DataStructs.TanimotoSimilarity(fp_mol_1, fp_mol_2)
    except Exception as e:
        if noException:
            return random.uniform(0.0, 1.0)
        raise Exception()


def fps_similarity_calc(fp_mol_1, fp_mol_2):
    try:
        return DataStructs.TanimotoSimilarity(fp_mol_1, fp_mol_2)
    except Exception as e:
        print("Exception has occurred in fps_similarity_calc")
        return random.uniform(0.0, 1.0)


# canonicalize_smiles
def canonicalize_smiles(molecule_SMILES):
    try:
        molecule_canonical_SMILES = Chem.MolToSmiles(Chem.MolFromSmiles(molecule_SMILES), True)
        return molecule_canonical_SMILES
    except Exception as e:
        raise Exception()


# valid if 1) valid by rdkit. 2) the property can be calculated on that molecule.
def is_valid_molecule(molecule_SMILES, property):
    try:
        property_calc(molecule_SMILES, property)
        smiles2fingerprint(molecule_SMILES)
        return True
    except Exception as e:
        return False


def calc_PL_score(molecule_SMILES):
    molecule_fp = smiles2fingerprint(molecule_SMILES)

    # patent_fp_tuple = (patent, fp)
    f_sim = lambda patent_fp_tuple: fps_similarity_calc(molecule_fp, patent_fp_tuple[1])
    return np.max(np.array([f_sim(patent_fp_tuple) for patent_fp_tuple in patentsFp_500000_list]))


def get_similar_patents_list(molecule_SMILES):
    # Density:
    patent = random.choices(list(density_500000_dict.keys()), weights=list(density_500000_dict.values()), k=1)[0]
    assert isinstance(patent, str)
    return patent

    # Closest:
    # if is_valid_molecule(molecule_SMILES, 'QED') and molecule_SMILES != '':
    #     molecule_fp = smiles2fingerprint(molecule_SMILES)
    #     # patent_fp_tuple = (patent, fp)
    #     f_sim = lambda patent_fp_tuple: fps_similarity_calc(molecule_fp, patent_fp_tuple[1])
    #     idx = np.argmax(np.array([f_sim(patent_fp_tuple) for patent_fp_tuple in patentsFp_500000_list]))
    #     patent = patentsFp_500000_list[idx][0]
    # else:
    #     patent = random.choices(list(density_500000_dict.keys()), weights=list(density_500000_dict.values()), k=1)[0]
    #
    # assert isinstance(patent, str)
    # return patent

    # k-closest:
    # if is_valid_molecule(molecule_SMILES, 'QED') and molecule_SMILES != '':
    #     patents_list = []
    #
    #     molecule_fp = smiles2fingerprint(molecule_SMILES)
    #     f_sim = lambda patent_fp_tuple: fps_similarity_calc(molecule_fp, patent_fp_tuple[1])
    #
    #     for patent_fp_tuple in patentsFp_500000_list:
    #         sim = f_sim(patent_fp_tuple)
    #         patents_list.append((patent_fp_tuple[0], sim))
    #
    #     patents_list.sort(key=lambda x: x[1], reverse=True)
    #     patents_list = patents_list[:10]
    #     patents_list = [patent for (patent, _) in patents_list]
    # else:
    #     patents_list = random.choices(list(density_500000_dict.keys()), weights=list(density_500000_dict.values()), k=10)
    #
    # assert isinstance(patents_list, list)
    # assert len(patents_list) == 10
    # return patents_list

    # Random:
    # patent = random.choices(list(density_500000_dict.keys()), k=1)[0]
    #
    # assert isinstance(patent, str)
    # return patent



