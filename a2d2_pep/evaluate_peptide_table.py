"""
Evaluate a finetuned peptide model checkpoint by sampling sequences
and computing metrics for the De Novo Peptide Generation table:
  Validity (%), Affinity (↑), Solubility (↑), Hemolysis (↑),
  Nonfouling (↑), Permeability (↑), Sampling Time (↓)
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
import pandas as pd

# add repo root (A2D2/) to sys.path so top-level packages like lightning_modules resolve
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from lightning_modules.any_length_remask import AnyOrderInsertionFlowModuleFT
from lightning_modules import AnyOrderInsertionFlowModule
from inference_quality import sample_peptides_eval
from pep_scoring.scoring_functions import ScoringFunctions
from pep_utils.analyzer import PeptideAnalyzer
from pep_scoring.tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from finetune_quality import PeptideFinetuner
from pep_utils.utils import str2bool, set_seed
from tdc import Evaluator


# Protein sequences
PROTEINS = {
    'amhr': 'MLGSLGLWALLPTAVEAPPNRRTCVFFEAPGVRGSTKTLGELLDTGTELPRAIRCLYSRCCFGIWNLTQDRAQVEMQGCRDSDEPGCESLHCDPSPRAHPSPGSTLFTCSCGTDFCNANYSHLPPPGSPGTPGSQGPQAAPGESIWMALVLLGLFLLLLLLLGSIILALLQRKNYRVRGEPVPEPRPDSGRDWSVELQELPELCFSQVIREGGHAVVWAGQLQGKLVAIKAFPPRSVAQFQAERALYELPGLQHDHIVRFITASRGGPGRLLSGPLLVLELHPKGSLCHYLTQYTSDWGSSLRMALSLAQGLAFLHEERWQNGQYKPGIAHRDLSSQNVLIREDGSCAIGDLGLALVLPGLTQPPAWTPTQPQGPAAIMEAGTQRYMAPELLDKTLDLQDWGMALRRADIYSLALLLWEILSRCPDLRPDSSPPPFQLAYEAELGNTPTSDELWALAVQERRRPYIPSTWRCFATDPDGLRELLEDCWDADPEARLTAECVQQRLAALAHPQESHPFPESCPRGCPPLCPEDCTSIPAPTILPCRPQRSACHFSVQQGPCSRNPQPACTLSPV',
    'tfr': 'MMDQARSAFSNLFGGEPLSYTRFSLARQVDGDNSHVEMKLAVDEEENADNNTKANVTKPKRCSGSICYGTIAVIVFFLIGFMIGYLGYCKGVEPKTECERLAGTESPVREEPGEDFPAARRLYWDDLKRKLSEKLDSTDFTGTIKLLNENSYVPREAGSQKDENLALYVENQFREFKLSKVWRDQHFVKIQVKDSAQNSVIIVDKNGRLVYLVENPGGYVAYSKAATVTGKLVHANFGTKKDFEDLYTPVNGSIVIVRAGKITFAEKVANAESLNAIGVLIYMDQTKFPIVNAELSFFGHAHLGTGDPYTPGFPSFNHTQFPPSRSSGLPNIPVQTISRAAAEKLFGNMEGDCPSDWKTDSTCRMVTSESKNVKLTVSNVLKEIKILNIFGVIKGFVEPDHYVVVGAQRDAWGPGAAKSGVGTALLLKLAQMFSDMVLKDGFQPSRSIIFASWSAGDFGSVGATEWLEGYLSSLHLKAFTYINLDKAVLGTSNFKVSASPLLYTLIEKTMQNVKHPVTGQFLYQDSNWASKVEKLTLDNAAFPFLAYSGIPAVSFCFCEDTDYPYLGTTMDTYKELIERIPELNKVARAAAEVAGQFVIKLTHDVELNLDYERYNSQLLSFVRDLNQYRADIKEMGLSLQWLYSARGDFFRATSRLTTDFGNAEKTDRFVMKKLNDRVMRVEYHFLSPYVSPKESPFRHVFWGSGSHTLPALLENLKLRKQNNGAFNETLFRNQLALATWTIQGAANALSGDVWDIDNEF',
    'gfap': 'MERRRITSAARRSYVSSGEMMVGGLAPGRRLGPGTRLSLARMPPPLPTRVDFSLAGALNAGFKETRASERAEMMELNDRFASYIEKVRFLEQQNKALAAELNQLRAKEPTKLADVYQAELRELRLRLDQLTANSARLEVERDNLAQDLATVRQKLQDETNLRLEAENNLAAYRQEADEATLARLDLERKIESLEEEIRFLRKIHEEEVRELQEQLARQQVHVELDVAKPDLTAALKEIRTQYEAMASSNMHEAEEWYRSKFADLTDAAARNAELLRQAKHEANDYRRQLQSLTCDLESLRGTNESLERQMREQEERHVREAASYQEALARLEEEGQSLKDEMARHLQEYQDLLNVKLALDIEIATYRKLLEGEENRITIPVQTFSNLQIRETSLDTKSVSEGHLKRNIVVKTVEMRDGEVIKESKQEHKDVM',
    'glp1': 'MAGAPGPLRLALLLLGMVGRAGPRPQGATVSLWETVQKWREYRRQCQRSLTEDPPPATDLFCNRTFDEYACWPDGEPGSFVNVSCPWYLPWASSVPQGHVYRFCTAEGLWLQKDNSSLPWRDLSECEESKRGERSSPEEQLLFLYIIYTVGYALSFSALVIASAILLGFRHLHCTRNYIHLNLFASFILRALSVFIKDAALKWMYSTAAQQHQWDGLLSYQDSLSCRLVFLLMQYCVAANYYWLLVEGVYLYTLLAFSVLSEQWIFRLYVSIGWGVPLLFVVPWGIVKYLYEDEGCWTRNSNMNYWLIIRLPILFAIGVNFLIFVRVICIVVSKLKANLMCKTDIKCRLAKSTLTLIPLLGTHEVIFAFVMDEHARGTLRFIKLFTELSFTSFQGLMVAILYCFVNNEVQLEFRKSWERWRLEHLHIQRDSSMKPLKCPTSSLSSGATAGSSMYTATCQASCS',
    'glast': 'MTKSNGEEPKMGGRMERFQQGVRKRTLLAKKKVQNITKEDVKSYLFRNAFVLLTVTAVIVGTILGFTLRPYRMSYREVKYFSFPGELLMRMLQMLVLPLIISSLVTGMAALDSKASGKMGMRAVVYYMTTTIIAVVIGIIIVIIIHPGKGTKENMHREGKIVRVTAADAFLDLIRNMFPPNLVEACFKQFKTNYEKRSFKVPIQANETLVGAVINNVSEAMETLTRITEELVPVPGSVNGVNALGLVVFSMCFGFVIGNMKEQGQALREFFDSLNEAIMRLVAVIMWYAPVGILFLIAGKIVEMEDMGVIGGQLAMYTVTVIVGLLIHAVIVLPLLYFLVTRKNPWVFIGGLLQALITALGTSSSSATLPITFKCLEENNGVDKRVTRFVLPVGATINMDGTALYEALAAIFIAQVNNFELNFGQIITISITATAASIGAAGIPQAGLVTMVIVLTSVGLPTDDITLIIAVDWFLDRLRTTTNVLGDSLGAGIVEHLSRHELKNRDVEMGNSVIEENEMKKPYQLIAQDNETEKPIDSETKM',
    'ncam': 'LQTKDLIWTLFFLGTAVSLQVDIVPSQGEISVGESKFFLCQVAGDAKDKDISWFSPNGEKLTPNQQRISVVWNDDSSSTLTIYNANIDDAGIYKCVVTGEDGSESEATVNVKIFQKLMFKNAPTPQEFREGEDAVIVCDVVSSLPPTIIWKHKGRDVILKKDVRFIVLSNNYLQIRGIKKTDEGTYRCEGRILARGEINFKDIQVIVNVPPTIQARQNIVNATANLGQSVTLVCDAEGFPEPTMSWTKDGEQIEQEEDDEKYIFSDDSSQLTIKKVDKNDEAEYICIAENKAGEQDATIHLKVFAKPKITYVENQTAMELEEQVTLTCEASGDPIPSITWRTSTRNISSEEKASWTRPEKQETLDGHMVVRSHARVSSLTLKSIQYTDAGEYICTASNTIGQDSQSMYLEVQYAPKLQGPVAVYTWEGNQVNITCEVFAYPSATISWFRDGQLLPSSNYSNIKIYNTPSASYLEVTPDSENDFGNYNCTAVNRIGQESLEFILVQADTPSSPSIDQVEPYSSTAQVQFDEPEATGGVPILKYKAEWRAVGEEVWHSKWYDAKEASMEGIVTIVGLKPETTYAVRLAALNGKGLGEISAASEF',
    'cereblon': 'MAGEGDQQDAAHNMGNHLPLLPAESEEEDEMEVEDQDSKEAKKPNIINFDTSLPTSHTYLGADMEEFHGRTLHDDDSCQVIPVLPQVMMILIPGQTLPLQLFHPQEVSMVRNLIQKDRTFAVLAYSNVQEREAQFGTTAEIYAYREEQDFGIEIVKVKAIGRQRFKVLELRTQSDGIQQAKVQILPECVLPSTMSAVQLESLNKCQIFPSKPVSREDQCSYKWWQKYQKRKFHCANLTSWPRWLYSLYDAETLMDRIKKQLREWDENLKDDSLPSNPIDFSYRVAACLPIDDVLRIQLLKIGSAIQRLRCELDIMNKCTSLCCKQCQETEITTKNEIFSLSLCGPMAAYVNPHGYVHETLTVYKACNLNLIGRPSTEHSWFPGYAWTVAQCKICASHIGWKFTATKKDMSPQKFWGLTRSALLPTIPDTEDEISPDKVILCL',
    'ligase': 'MASQPPEDTAESQASDELECKICYNRYNLKQRKPKVLECCHRVCAKCLYKIIDFGDSPQGVIVCPFCRFETCLPDDEVSSLPDDNNILVNLTCGGKGKKCLPENPTELLLTPKRLASLVSPSHTSSNCLVITIMEVQRESSPSLSSTPVVEFYRPASFDSVTTVSHNWTVWNCTSLLFQTSIRVLVWLLGLLYFSSLPLGIYLLVSKKVTLGVVFVSLVPSSLVILMVYGFCQCVCHEFLDCMAPPS',
    'skp2': 'MHRKHLQEIPDLSSNVATSFTWGWDSSKTSELLSGMGVSALEKEEPDSENIPQELLSNLGHPESPPRKRLKSKGSDKDFVIVRRPKLNRENFPGVSWDSLPDELLLGIFSCLCLPELLKVSGVCKRWYRLASDESLWQTLDLTGKNLHPDVTGRLLSQGVIAFRCPRSFMDQPLAEHFSPFRVQHMDLSNSVIEVSTLHGILSQCSKLQNLSLEGLRLSDPIVNTLAKNSNLVRLNLSGCSGFSEFALQTLLSSCSRLDELNLSWCFDFTEKHVQVAVAHVSETITQLNLSGYRKNLQKSDLSTLVRRCPNLVHLDLSDSVMLKNDCFQEFFQLNYLQHLSLSRCYDIIPETLLELGEIPTLKTLQVFGIVPDGTLQLLKEALPHLQINCSHFTTIARPTIGNKKNQEIWGIKCRLTLQKPSCL',
}


def load_finetuned_model(checkpoint_path, pretrained_ckpt_path, device='cuda'):
    """Load a finetuned PeptideFinetuner from a Lightning checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hparams = ckpt.get('hyper_parameters', {})
    args = hparams.get('args', None)

    # Load pretrained base checkpoint to get config
    base_ckpt = torch.load(pretrained_ckpt_path, map_location='cpu', weights_only=False)
    if 'hyper_parameters' in base_ckpt:
        config = base_ckpt['hyper_parameters']['config']
    elif 'config' in base_ckpt:
        config = base_ckpt['config']
    else:
        raise ValueError("Cannot find config in base checkpoint")

    from omegaconf import OmegaConf, DictConfig
    if not OmegaConf.is_config(config):
        config = DictConfig(config)
    OmegaConf.set_struct(config, False)

    config.training.use_adaptive_schedule = getattr(args, 'use_adaptive_schedule', True)
    config.training.schedule_hidden_dim = getattr(args, 'schedule_hidden_dim', 256)
    config.training.schedule_num_layers = getattr(args, 'schedule_num_layers', 2)
    config.training.schedule_loss_weight = getattr(args, 'schedule_loss_weight', 0.1)
    config.training.freeze_base_model = getattr(args, 'freeze_base_model', False)
    config.training.schedule_warmup_epochs = getattr(args, 'schedule_warmup_epochs', 0)
    OmegaConf.set_struct(config, True)

    disable_planner = getattr(args, 'disable_planner', False)

    policy_model = AnyOrderInsertionFlowModuleFT(
        config=config,
        args=args,
        pretrained_checkpoint=pretrained_ckpt_path,
        insertion_planner=not disable_planner,
    )

    # Load finetuned weights
    state_dict = ckpt['state_dict']
    policy_state = {}
    for k, v in state_dict.items():
        if k.startswith('policy_model.'):
            policy_state[k[len('policy_model.'):]] = v
    policy_model.load_state_dict(policy_state, strict=False)
    policy_model = policy_model.to(device)
    policy_model.eval()

    return policy_model, args, config


@torch.no_grad()
def evaluate_checkpoint(policy_model, tokenizer, reward_model, analyzer,
                        num_samples=1000, batch_size=50, max_length=512,
                        total_num_steps=256, quality_mode="both", num_remasking=3,
                        quality_threshold=0.5, unmask_quality_threshold=None, device='cuda'):
    """
    Sample `num_samples` peptides and compute all table metrics.
    Returns a dict with: validity, affinity, sol, hemo, nf, permeability, sampling_time
    """
    all_affinity = []
    all_sol = []
    all_hemo = []
    all_nf = []
    all_permeability = []
    all_valid_seqs = []
    total_valid = 0
    total_generated = 0
    total_time = 0.0

    num_batches = (num_samples + batch_size - 1) // batch_size
    remaining = num_samples

    for b in range(num_batches):
        bs = min(batch_size, remaining)
        remaining -= bs

        t_start = time.time()
        result = sample_peptides_eval(
            model=policy_model,
            reward_model=reward_model,
            analyzer=analyzer,
            tokenizer=tokenizer,
            steps=total_num_steps,
            mask=policy_model.interpolant.mask_token,
            pad=policy_model.interpolant.pad_token,
            batch_size=bs,
            max_length=max_length,
            quality_mode=quality_mode,
            num_remasking=num_remasking,
            quality_threshold=quality_threshold,
            unmask_quality_threshold=unmask_quality_threshold,
            return_valid=True,
        )
        t_end = time.time()

        # Unpack: validSequences, affinity, sol, hemo, nf, permeability, valid_fraction
        valid_seqs, affinity, sol, hemo, nf, permeability, valid_fraction = result

        batch_valid = len(valid_seqs)
        total_valid += batch_valid
        total_generated += bs
        total_time += (t_end - t_start)
        all_valid_seqs.extend(valid_seqs)

        if isinstance(affinity, (list, np.ndarray)) and len(affinity) > 0:
            all_affinity.extend(affinity if isinstance(affinity, list) else affinity.tolist())
            all_sol.extend(sol if isinstance(sol, list) else sol.tolist())
            all_hemo.extend(hemo if isinstance(hemo, list) else hemo.tolist())
            all_nf.extend(nf if isinstance(nf, list) else nf.tolist())
            all_permeability.extend(permeability if isinstance(permeability, list) else permeability.tolist())

        print(f"  Batch {b+1}/{num_batches}: {batch_valid}/{bs} valid, "
              f"time={t_end - t_start:.1f}s")

    validity = total_valid / total_generated * 100.0 if total_generated > 0 else 0.0

    # Uniqueness (% of valid sequences that are unique) and
    # Diversity (1 - mean pairwise Tanimoto on Morgan FPs of unique sequences).
    # Matches the convention used in evaluate_mol_table.py.
    all_unique = list(set(all_valid_seqs))
    num_unique = len(all_unique)
    uniqueness = num_unique / total_valid * 100.0 if total_valid > 0 else 0.0
    if num_unique > 1:
        diversity = Evaluator('diversity')(all_unique)
    else:
        diversity = 0.0

    metrics = {
        'Validity (%)': validity,
        'Uniqueness (%)': uniqueness,
        'Diversity': diversity,
        'Affinity': np.mean(all_affinity) if all_affinity else 0.0,
        'Affinity Std': np.std(all_affinity) if all_affinity else 0.0,
        'Solubility': np.mean(all_sol) if all_sol else 0.0,
        'Solubility Std': np.std(all_sol) if all_sol else 0.0,
        'Hemolysis': np.mean(all_hemo) if all_hemo else 0.0,
        'Hemolysis Std': np.std(all_hemo) if all_hemo else 0.0,
        'Nonfouling': np.mean(all_nf) if all_nf else 0.0,
        'Nonfouling Std': np.std(all_nf) if all_nf else 0.0,
        'Permeability': np.mean(all_permeability) if all_permeability else 0.0,
        'Permeability Std': np.std(all_permeability) if all_permeability else 0.0,
        'Sampling Time (s)': total_time,
        'Num Generated': total_generated,
        'Num Valid': total_valid,
        'Num Unique': num_unique,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a finetuned peptide checkpoint")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the finetuned Lightning checkpoint (e.g., last.ckpt)')
    parser.add_argument('--pretrained_ckpt', type=str,
                        default=os.path.join(REPO_ROOT, 'pretrained', 'anylength_pep.ckpt'),
                        help='Path to the pretrained base model checkpoint')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of peptides to sample')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for sampling')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--total_num_steps', type=int, default=256)
    parser.add_argument('--num_remasking', type=int, default=3)
    parser.add_argument('--quality_threshold', type=float, default=0.5,
                        help='Threshold for insertion quality filtering during sampling')
    parser.add_argument('--unmask_quality_threshold', type=float, default=None,
                        help='If set, gate unmasking/remasking by confidence: remask '
                             'ALL clean tokens whose unmasking confidence is below this '
                             'threshold, regardless of the schedule budget. If unset '
                             '(default), remasking is purely schedule-driven (count-based).')
    parser.add_argument('--prot_name', type=str, default='glast',
                        help='Target protein name (must be one of: ' + ', '.join(PROTEINS.keys()) + ')')
    parser.add_argument('--prot_seq', type=str, default=None,
                        help='Custom protein sequence (overrides --prot_name)')
    parser.add_argument('--disable_planner', action='store_true',
                        help='If set, disable remasking during evaluation')
    parser.add_argument('--disable_insertion_planner', action='store_true',
                        help='If set, disable insertion quality filtering during evaluation')
    parser.add_argument('--disable_unmasking_planner', action='store_true',
                        help='If set, disable unmasking confidence planner during evaluation')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results CSV. Defaults to checkpoint directory.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed, use_cuda=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Map flags to quality_mode
    if args.disable_planner:
        quality_mode = "none"
    elif args.disable_insertion_planner and args.disable_unmasking_planner:
        quality_mode = "none"
    elif args.disable_insertion_planner:
        quality_mode = "unmasking_only"
    elif args.disable_unmasking_planner:
        quality_mode = "insertion_only"
    else:
        quality_mode = "both"

    print(f"Loading checkpoint: {args.checkpoint_path}")
    print(f"Pretrained base: {args.pretrained_ckpt}")
    print(f"Quality mode: {quality_mode}")

    policy_model, train_args, config = load_finetuned_model(
        args.checkpoint_path, args.pretrained_ckpt, device=device
    )

    # Setup tokenizer, reward model, analyzer
    tokenizer = SMILES_SPE_Tokenizer(
        os.path.join(REPO_ROOT, 'a2d2_pep', 'pep_scoring', 'tokenizer', 'new_vocab.txt'),
        os.path.join(REPO_ROOT, 'a2d2_pep', 'pep_scoring', 'tokenizer', 'new_splits.txt')
    )

    if args.prot_seq is not None:
        prot = args.prot_seq
        prot_name = args.prot_name
    else:
        prot_name = args.prot_name
        if prot_name not in PROTEINS:
            raise ValueError(f"Unknown protein: {prot_name}. Choose from: {list(PROTEINS.keys())}")
        prot = PROTEINS[prot_name]

    score_func_names = ['binding_affinity1', 'solubility', 'hemolysis', 'nonfouling', 'permeability']
    reward_model = ScoringFunctions(score_func_names, prot_seqs=[prot], device=device)
    analyzer = PeptideAnalyzer()

    print(f"\nSampling {args.num_samples} peptides (quality_mode={quality_mode}, target={prot_name})...")

    metrics = evaluate_checkpoint(
        policy_model=policy_model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        analyzer=analyzer,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        total_num_steps=args.total_num_steps,
        quality_mode=quality_mode,
        num_remasking=args.num_remasking,
        quality_threshold=args.quality_threshold,
        unmask_quality_threshold=args.unmask_quality_threshold,
        device=device,
    )

    # Print summary table
    print("\n" + "=" * 60)
    print("  De Novo Peptide Generation Results")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30s}: {v:.4f}")
        else:
            print(f"  {k:<30s}: {v}")
    print("=" * 60)

    # Save results
    output_dir = args.output_dir or os.path.dirname(args.checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    if args.disable_planner:
        tag = "no_planner"
    elif args.disable_insertion_planner:
        tag = "no_insertion_planner"
    elif args.disable_unmasking_planner:
        tag = "no_unmasking_planner"
    else:
        tag = "with_planner"
    if args.unmask_quality_threshold is not None:
        tag += f"_ut{args.unmask_quality_threshold:g}"
    # Record the sweep parameter in the saved row for traceability.
    metrics['unmask_quality_threshold'] = args.unmask_quality_threshold
    metrics['quality_threshold'] = args.quality_threshold
    metrics_path = os.path.join(output_dir, f'eval_metrics_{tag}_{prot_name}.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()
