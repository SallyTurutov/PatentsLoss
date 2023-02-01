import torch
import pandas as pd

from model import RNN
from data_structs import Vocabulary
from utils import seq_to_smiles, unique

if __name__ == "__main__":
    voc = Vocabulary(init_from_file="data/Voc")

    Prior = RNN(voc)
    Agent = RNN(voc)

    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load('data/Prior.ckpt'))
        Agent.rnn.load_state_dict(torch.load('data/Agent.ckpt'))
    else:
        Prior.rnn.load_state_dict(torch.load('data/Prior.ckpt', map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load('data/Agent.ckpt', map_location=lambda storage, loc: storage))

    # Sample from Agent
    seqs, agent_likelihood, entropy = Agent.sample(800)

    # Remove duplicates, ie only consider unique seqs
    unique_idxs = unique(seqs)
    seqs = seqs[unique_idxs]
    agent_likelihood = agent_likelihood[unique_idxs]
    entropy = entropy[unique_idxs]

    # Get prior likelihood and score
    prior_likelihood, _ = Prior.likelihood(seqs)
    smiles = seq_to_smiles(seqs, voc)

    results = []
    for mol_smiles in smiles:
        results.append((mol_smiles, mol_smiles))

    results = pd.DataFrame(results)
    results.to_csv('REINVEN_QED_Random_test.txt', index=False, header=False, sep=' ')