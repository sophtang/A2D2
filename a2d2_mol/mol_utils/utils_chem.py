# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import random
import safe as sf
import datamol as dm
from contextlib import suppress
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# https://github.com/datamol-io/safe/blob/main/safe/sample.py
# https://github.com/jensengroup/GB_GA/blob/master/crossover.py
def safe_to_smiles(safe_str, fix=True):
    if fix:
        safe_str = '.'.join([frag for frag in safe_str.split('.')
                             if sf.decode(frag, ignore_errors=True) is not None])
    return sf.decode(safe_str, canonical=True, ignore_errors=True)


def _safe_to_smiles_worker(args):
    """Worker function for parallel SAFE to SMILES conversion."""
    safe_str, use_bracket_safe, fix = args
    try:
        from mol_utils.bracket_safe_converter import bracketsafe2safe
        if use_bracket_safe:
            safe_str = bracketsafe2safe(safe_str)
        return safe_to_smiles(safe_str, fix=fix)
    except Exception:
        return None


def batch_safe_to_smiles(safe_strings, use_bracket_safe=False, fix=True, num_workers=None):
    """
    Convert a batch of SAFE strings to SMILES in parallel using multiprocessing.
    
    Args:
        safe_strings: List of SAFE format strings
        use_bracket_safe: Whether to convert from bracket SAFE format first
        fix: Whether to fix invalid fragments
        num_workers: Number of parallel workers (default: min(cpu_count, len(safe_strings), 8))
    
    Returns:
        List of SMILES strings (None for invalid molecules)
    """
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    import os
    
    n = len(safe_strings)
    if n == 0:
        return []
    
    # For small batches, use sequential processing (overhead not worth it)
    if n <= 4:
        if use_bracket_safe:
            from mol_utils.bracket_safe_converter import bracketsafe2safe
            return [safe_to_smiles(bracketsafe2safe(s), fix=fix) for s in safe_strings]
        else:
            return [safe_to_smiles(s, fix=fix) for s in safe_strings]
    
    # Use ThreadPoolExecutor for I/O bound tasks (RDKit releases GIL)
    # ProcessPoolExecutor has too much overhead for this use case
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, n, 8)
    
    args_list = [(s, use_bracket_safe, fix) for s in safe_strings]
    
    # ThreadPoolExecutor is faster here because:
    # 1. No pickle serialization overhead
    # 2. RDKit releases the GIL during computation
    # 3. Lower startup cost
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_safe_to_smiles_worker, args_list))
    
    return results


def batch_validate_and_extract(smiles_list, samples_tensor, log_rnd_tensor):
    """
    Batch validate SMILES and extract valid samples efficiently.
    
    Args:
        smiles_list: List of SMILES strings (may contain None for invalid)
        samples_tensor: Tensor of token IDs (B, L)
        log_rnd_tensor: Tensor of log random values (B,)
    
    Returns:
        valid_sequences: List of valid SMILES (largest fragment)
        valid_indices: List of indices of valid samples
    """
    valid_sequences = []
    valid_indices = []
    
    for idx, smiles in enumerate(smiles_list):
        if smiles:  # Valid SMILES
            # Take largest fragment if multiple
            largest_fragment = sorted(smiles.split('.'), key=len)[-1]
            valid_sequences.append(largest_fragment)
            valid_indices.append(idx)
    
    return valid_sequences, valid_indices


def filter_by_substructure(sequences, substruct):
    substruct = sf.utils.standardize_attach(substruct)
    substruct = Chem.DeleteSubstructs(Chem.MolFromSmarts(substruct), Chem.MolFromSmiles('*'))
    substruct = Chem.MolFromSmarts(Chem.MolToSmiles(substruct))
    return sf.utils.filter_by_substructure_constraints(sequences, substruct)


def mix_sequences(prefix_sequences, suffix_sequences, prefix, suffix, num_samples=1):
    mol_linker_slicer = sf.utils.MolSlicer(require_ring_system=False)

    prefix_linkers = []
    suffix_linkers = []
    prefix_query = dm.from_smarts(prefix)
    suffix_query = dm.from_smarts(suffix)

    for x in prefix_sequences:
        with suppress(Exception):
            x = dm.to_mol(x)
            out = mol_linker_slicer(x, prefix_query)
            prefix_linkers.append(out[1])

    for x in suffix_sequences:
        with suppress(Exception):
            x = dm.to_mol(x)
            out = mol_linker_slicer(x, suffix_query)
            suffix_linkers.append(out[1])

    n_linked = 0
    linked = []
    linkers = prefix_linkers + suffix_linkers
    linkers = [x for x in linkers if x is not None]
    for n_linked, linker in enumerate(linkers):
        linked.extend(mol_linker_slicer.link_fragments(linker, prefix, suffix))
        if n_linked > num_samples:
            break
        linked = [x for x in linked if x]
    return linked[:num_samples]
    

def cut(smiles):
    def cut_nonring(mol):
        if not mol.HasSubstructMatch(Chem.MolFromSmarts('[*]-;!@[*]')):
            return None

        bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]')))  # single bond not in ring
        bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]
        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

        try:
            return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
        except ValueError:
            return None
        
    mol = Chem.MolFromSmiles(smiles)
    frags = set()
    # non-ring cut
    for _ in range(3):
        frags_nonring = cut_nonring(mol)
        if frags_nonring is not None:
            frags |= set([Chem.MolToSmiles(f) for f in frags_nonring])
    return frags


class Slicer:
    def __call__(self, mol):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        # non-ring single bonds
        bonds = mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]'))
        for bond in bonds:
            yield bond