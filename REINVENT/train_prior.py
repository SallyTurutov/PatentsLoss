#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from data_structs import MolData, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate, seq_to_smiles
from property_handler import property_init, get_similar_patents_list

rdBase.DisableLog('rdApp.error')


def pretrain(voc, restore_from=None):
    """Trains the Prior RNN"""

    # Read vocabulary from a file
    # voc = Vocabulary(init_from_file="data/Voc")

    # Create a Dataset from a SMILES file
    moldata = MolData("data/zinc_filtered", voc)
    data = DataLoader(moldata, batch_size=128, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    Prior = RNN(voc)

    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr=0.001)
    for epoch in range(1, 10):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            seqs = batch.long()
            smiles = seq_to_smiles(seqs, voc)
            patents_lists = [get_similar_patents_list(molecule_SMILES) for molecule_SMILES in smiles]
            patents_loss_list = []
            for idx in range(len(patents_lists[0])):
                patents = []
                for patents_list in patents_lists:
                    patents.append(patents_list[idx])

                    moldata_patents = MolData(patents, voc, isFile=False)
                    patents_data = DataLoader(moldata_patents, batch_size=128, shuffle=True, drop_last=True, collate_fn=MolData.collate_fn)
                    for patents_batch in patents_data:
                        patents_seqs = patents_batch.long()
                        patents_log_p, _ = Prior.likelihood(patents_seqs)
                        patents_loss = - patents_log_p.mean()
                        patents_loss_list.append(patents_loss)


            # Calculate loss
            log_p, _ = Prior.likelihood(seqs)
            loss = - log_p.mean()

            patents_loss = torch.log(torch.mean(torch.stack(patents_loss_list)))
            final_loss = loss - patents_loss

            # Calculate gradients and take a step
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            if step % 500 == 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    final_loss: {:5.2f}   loss: {:5.2f}   patents_loss: {:5.2f}\n".format(epoch, step, final_loss.item(), loss.item(), patents_loss.item()))
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")
                torch.save(Prior.rnn.state_dict(), "data/Prior.ckpt")

        # Save the Prior
        torch.save(Prior.rnn.state_dict(), "data/Prior.ckpt")


if __name__ == "__main__":
    voc = Vocabulary(init_from_file="data/Voc")
    property_init(voc)
    pretrain(voc)
