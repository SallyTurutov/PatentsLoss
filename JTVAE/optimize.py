import torch
import rdkit

import pandas as pd
import argparse
from tqdm import tqdm
from datautils import *
from jtnn_vae import *
from vocab import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', required=True)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

df = pd.read_csv('data/zinc/test.txt', header=None)
testset = list(df.iloc[:, 0])
torch.manual_seed(0)

model_ckpts = [fn for fn in os.listdir('save_model')]
for model_ckpt in model_ckpts:
    print('Current Checkpoint: ' + str(model_ckpt))
    model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)
    model.load_state_dict(torch.load('save_model/' + str(model_ckpt)))
    model = model.cuda()

    results = []
    loader = MolTreeFolder('data/test', vocab, 1, num_workers=4, shuffle=False)
    for idx, batch in tqdm(enumerate(loader)):
        if idx == 800:
            break
        in_smiles = testset[idx]
        if batch is None:
            out_smiles = ''
        else:
            out_smiles = model.predict(batch)
        results.append((in_smiles, out_smiles))

    results = pd.DataFrame(results)
    results.to_csv('data/test/' + str(model_ckpt) + '-results.txt', index=False, header=False, sep=' ')