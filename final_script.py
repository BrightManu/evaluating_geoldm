# Import libraries
import os
import json
import math
import random
import pickle
import numpy as np
import torch
from rdkit import RDLogger, Chem
from rdkit.Chem import QED, rdMolDescriptors, SDWriter, Draw, rdmolfiles, GetPeriodicTable
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from torch.distributions import Categorical
from scipy.stats import wasserstein_distance

# Set seed
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

RDLogger.DisableLog('rdApp.warning')

# Convert latent samples to RDKit molecules with bond prediction

def to_rdkit_with_bond_prediction(samples):
    mols = []
    for s in samples:
        z, pos, mask = s['z'], s['pos'], s['mask']
        n = int(mask.sum())
        mol = Chem.RWMol()
        for atom_z in z[:n]:
            mol.AddAtom(Chem.Atom(int(atom_z)))
        conf = Chem.Conformer(n)
        for i in range(n):
            x, y, zc = map(float, pos[i])
            conf.SetAtomPosition(i, (x, y, zc))
        mol.AddConformer(conf)
        try:
            mb = Chem.MolToMolBlock(mol)
            m2 = rdmolfiles.MolFromMolBlock(mb, sanitize=True, removeHs=False)
            if m2 is not None:
                mols.append(m2)
        except:
            continue
    return mols

# Atom stability: fraction of atoms with correct total valence

def compute_atom_stability(mols):
    ok = total = 0
    for m in mols:
        for atom in m.GetAtoms():
            total += 1
            if atom.GetTotalValence() == atom.GetFormalCharge() + atom.GetDegree():
                ok += 1
    return (ok / total * 100) if total else 0

# Compute distributions for QED and atom counts

def compute_props(mols):
    props = {'qed': [], 'atoms': []}
    for m in mols:
        try:
            props['qed'].append(QED.qed(m))
            props['atoms'].append(m.GetNumAtoms())
        except:
            pass
    return props

# Subsample real props to match gen size

def subsample_props(real_props, n_gen, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(real_props['qed']), size=n_gen, replace=False)
    return {k: [real_props[k][i] for i in idx] for k in real_props}

# Wasserstein distances

def compute_wd(gen_props, real_props):
    return {
        'WD_QED': float(wasserstein_distance(gen_props['qed'], real_props['qed'])),
        'WD_Atoms': float(wasserstein_distance(gen_props['atoms'], real_props['atoms'])),
    }

# ---- Main ----
def main():
    set_seed(42)

    # Load dataset
    data = np.load('e-drugs_conf.npy')
    mol_ids = data[:,0].astype(int)
    atom_nums = data[:,1].astype(int)

    # Build priors
    # atom type prior
    unique_z = sorted({int(z) for z in data[:,1]})
    counts_z = [(atom_nums == z).sum() for z in unique_z]
    prop_dist = Categorical(probs=torch.tensor(counts_z, dtype=torch.float32))
    # node count prior
    counts = [(mol_ids == mid).sum() for mid in np.unique(mol_ids)]
    hist = np.bincount(counts, minlength=max(counts)+1)[:max(counts)+1]
    nodes_dist = Categorical(probs=torch.tensor(hist, dtype=torch.float32))

    # Element frequency plot
    stats_dir = 'stats'
    os.makedirs(stats_dir, exist_ok=True)
    pt = GetPeriodicTable()
    elements = [pt.GetElementSymbol(int(z)) for z in unique_z]
    plt.figure(figsize=(8,5))
    sns.barplot(x=elements, y=counts_z)
    plt.title('Element Frequency')
    plt.savefig(os.path.join(stats_dir, 'element_frequency.png'))
    plt.close()

    # Load pretrained model
    args_path  = '/home/bkmanu/GenAI/Project/drugs_latent2/args.pickle'
    model_path = '/home/bkmanu/GenAI/Project/drugs_latent2/generative_model.npy'
    with open(args_path, 'rb') as f:
        args = pickle.load(f)
    args.dataset  = 'e_drugs'
    args.remove_h = False
    args.device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # extend path to GeoLDM
    import sys
    sys.path.insert(0, os.path.join(os.getcwd(), 'GeoLDM'))
    from configs.datasets_config import get_dataset_info
    from qm9.models import get_latent_diffusion
    from qm9.sampling import sample
    dataset_info = get_dataset_info('e_drugs', False)
    # pad to 16 atom types
    while len(dataset_info['atom_decoder']) < 16:
        idx = len(dataset_info['atom_decoder'])
        pad = f'X{idx}'
        dataset_info['atom_decoder'].append(pad)
        dataset_info['atom_encoder'][pad] = idx
        dataset_info['atom_types'][idx] = 0
        dataset_info['colors_dic'].append('C7')
        dataset_info['radius_dic'].append(0.6)

    model, _, _ = get_latent_diffusion(args, args.device, dataset_info, None)
    state = torch.load(model_path, map_location=args.device)
    model.load_state_dict(state)
    model.eval()

    # Prepare real molecule objects for metrics
    real_mols = []
    for mid in np.unique(mol_ids):
        block = data[mol_ids == mid]
        m = Chem.RWMol()
        for z in block[:,1]: m.AddAtom(Chem.Atom(int(z)))
        conf = Chem.Conformer(len(block))
        for i,(x,y,z) in enumerate(block[:,2:]): conf.SetAtomPosition(i,(float(x),float(y),float(z)))
        m.AddConformer(conf)
        try:
            Chem.SanitizeMol(m)
            real_mols.append(m)
        except:
            pass
    real_props = compute_props(real_mols)

    # Sampling
    sample_configs = [(10,2), (100,10), (1000,10)]
    for n_samples, batch_size in sample_configs:
        results_dir = f'results_{n_samples}'
        os.makedirs(results_dir, exist_ok=True)
        generated = []

        for i in tqdm(range(math.ceil(n_samples/batch_size)), desc=f'Sampling {n_samples}'):
            bs = min(batch_size, n_samples - len(generated))
            nodes = nodes_dist.sample((bs,)).to(args.device)
            try:
                one_hot, charges, xyz, node_mask = sample(
                    args, args.device, model, dataset_info,
                    nodesxsample=nodes, prop_dist=prop_dist
                )
                for j in range(bs):
                    z = one_hot[j].argmax(dim=-1).cpu().numpy()
                    pos = xyz[j].cpu().numpy()
                    mask = node_mask[j].cpu().numpy()
                    generated.append({'z':z,'pos':pos,'mask':mask})
            except AssertionError:
                continue
            if len(generated) >= n_samples:
                break
        generated = generated[:n_samples]
        np.save(os.path.join(results_dir,'latent.npy'), np.array(generated,object))

        # Convert and predict bonds
        gen_mols = to_rdkit_with_bond_prediction(generated)
        atom_stab = compute_atom_stability(gen_mols)

        # save generated molecules in SDF file
        writer = SDWriter(os.path.join(results_dir,'generated_3d.sdf'))
        for m in gen_mols: writer.write(m)
        writer.close()

        # save SMILES
        with open(os.path.join(results_dir,'gen_smiles.txt'),'w') as f:
            for m in gen_mols: f.write(Chem.MolToSmiles(m)+'\n')

        # Compute props and Wasserstein Distance
        gen_props = compute_props(gen_mols)
        real_sub = subsample_props(real_props, len(gen_props['qed']))
        wd = compute_wd(gen_props, real_sub)

        # Plot KDEs
        for key in gen_props:
            plt.figure()
            if real_sub[key]: sns.kdeplot(real_sub[key],label='Real',fill=True)
            if gen_props[key]: sns.kdeplot(gen_props[key],label='Gen',fill=True)
            plt.title(f'{key} dist')
            plt.legend(); plt.savefig(os.path.join(results_dir,f'kde_{key}.png')); plt.close()

        # Metrics
        num_valid = len(gen_mols)
        num_unique = len(set(Chem.MolToSmiles(m) for m in gen_mols))
        num_novel = len(set(Chem.MolToSmiles(m) for m in gen_mols) - set(Chem.MolToSmiles(m) for m in real_mols))
        metrics = {
            'validity': 100 * num_valid / n_samples,
            'uniqueness': 100 * num_unique / num_valid if num_valid else 0,
            'novelty': 100 * num_novel / num_unique if num_unique else 0,
            'atom_stability': atom_stab,
            **wd
        }
        with open(os.path.join(results_dir,'metrics.json'),'w') as f:
            json.dump(metrics,f,indent=2)

        print(f"Done {n_samples}:", metrics)

if __name__ == '__main__':
    main()