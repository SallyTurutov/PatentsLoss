import pandas as pd
import pickle
import datetime
import random
import torch.multiprocessing as mp

# my files
from common_utils import set_seed
from property_handler import fps_similarity_calc, rdkit_no_error_print, smiles2fingerprint

if __name__ == "__main__":
    rdkit_no_error_print()

    patents_path = 'dataset/PL/SureChEMBL.txt'
    density_path = 'dataset/PL/densityProbs500000.pkl'
    patentsFp_path = 'dataset/PL/patentsFp.pkl'

    epoch_init = 0
    num_of_GPU_cores = 1

    # set seed
    seed = 50
    set_seed(seed)

    patents_df = pd.read_csv(patents_path, header=None)

    print('Make yourself a coffee, it will take a while...')


    def calc_fp_helper(patent):
        return patent, smiles2fingerprint(patent)



    tmp_patent_fp_list = random.choices(list(patents_df.iloc[:, 0]), k=500000)
    with mp.Pool() as pool:
        patent_fp_list = pool.map(calc_fp_helper, tmp_patent_fp_list)

    patentlist_data = {}
    for (patent, fp) in patent_fp_list: 
        patentlist_data[patent] = fp

    patentsFp_file = open(patentsFp_path, "wb")
    pickle.dump(patentlist_data, patentsFp_file)
    patentsFp_file.close()

    print("Done preparing patent_fp_list")

    def calc_sim_helper(input_patent_fp_tuple):
        print('Starting ' + str(input_patent_fp_tuple) + " at " + str(datetime.datetime.now()))
        # patent_fp_list is given
        # patent_fp_tuple = (patent, fp)

        counter = 0
        patent = input_patent_fp_tuple[0]
        fp = input_patent_fp_tuple[1]

        for patent_fp_tuple in patent_fp_list:
            counter += fps_similarity_calc(fp, patent_fp_tuple[1])
        return patent, counter


    heuristics_dict = {}

    with mp.Pool() as pool:
        patents_heuristics_list = pool.map(calc_sim_helper, patent_fp_list)

    for patent, sim_val in patents_heuristics_list:
        heuristics_dict[patent] = sim_val

    total = sum(heuristics_dict.values())
    heuristics_data = {k: v / total for k, v in heuristics_dict.items()}

    density_file = open(density_path, "wb")
    pickle.dump(heuristics_data, density_file)
    density_file.close()

