import os
import yaml
import logging
import json
import torch
import configargparse
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from dreamsim import PerceptualModel
from dataset.dataset import TwoAFCDataset
from training.train import LightningPerceptualModel
from evaluation.score import score_nights_dataset, score_things_dataset, score_bapps_dataset
from evaluation.eval_datasets import ThingsDataset, BAPPSDataset

log = logging.getLogger("lightning.pytorch")
log.propagate = False
log.setLevel(logging.ERROR)

def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')

    ## Run options
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', type=str, default="./eval_outputs", help="Dir to save results in.")
    parser.add_argument('--tag', type=str, default="", help="Exp name for saving results")
    

    ## Checkpoint evaluation options
    parser.add_argument('--eval_checkpoint', type=str, help="Path to a checkpoint root.")
    parser.add_argument('--eval_checkpoint_cfg', type=str, help="Path to checkpoint config.")
    parser.add_argument('--load_dir', type=str, default="./models", help='path to pretrained ViT checkpoints.')

    ## Baseline evaluation options
    parser.add_argument('--baseline_model', type=str, default=None)
    parser.add_argument('--baseline_feat_type', type=str, default=None)
    parser.add_argument('--baseline_stride', type=str, default=None)

    ## Dataset options
    parser.add_argument('--nights_root', type=str, default=None, help='path to nights dataset.')
    parser.add_argument('--bapps_root', type=str, default=None, help='path to bapps images.')
    parser.add_argument('--things_root', type=str, default=None, help='path to things images.')
    parser.add_argument('--things_file', type=str, default=None, help='path to things info file.')
    parser.add_argument('--df2_root', type=str, default=None, help='path to df2 root.')
    parser.add_argument('--df2_gt', type=str, default=None, help='path to df2 ground truth json.')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4, help='dataset batch size.')

    return parser.parse_args()

def load_dreamsim_model(args, device="cuda"):
    with open(os.path.join(args.eval_checkpoint_cfg), "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    model_cfg = cfg
    model_cfg['load_dir'] = args.load_dir
    model = LightningPerceptualModel(**model_cfg)
    model.load_lora_weights(args.eval_checkpoint)
    model = model.perceptual_model.to(device)
    preprocess = "DEFAULT"
    return model, preprocess


def load_baseline_model(args, device="cuda"):
    model = PerceptualModel(model_type=args.baseline_model, feat_type=args.baseline_feat_type, stride=args.baseline_stride, baseline=True, load_dir=args.load_dir)
    model = model.to(device)
    preprocess = "DEFAULT"
    return model, preprocess

def eval_nights(model, preprocess, device, args):
    eval_results = {}
    val_dataset = TwoAFCDataset(root_dir=args.nights_root, split="val", preprocess=preprocess)
    test_dataset_imagenet = TwoAFCDataset(root_dir=args.nights_root, split="test_imagenet", preprocess=preprocess)
    test_dataset_no_imagenet = TwoAFCDataset(root_dir=args.nights_root, split="test_no_imagenet", preprocess=preprocess)
    total_length = len(test_dataset_no_imagenet) + len(test_dataset_imagenet)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_imagenet_loader = DataLoader(test_dataset_imagenet, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_no_imagenet_loader = DataLoader(test_dataset_no_imagenet, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    val_score = score_nights_dataset(model, val_loader, device)
    imagenet_score = score_nights_dataset(model, test_imagenet_loader, device)
    no_imagenet_score = score_nights_dataset(model, test_no_imagenet_loader, device)

    eval_results['nights_val'] = val_score.item()
    eval_results['nights_imagenet'] = imagenet_score.item()
    eval_results['nights_no_imagenet'] = no_imagenet_score.item()
    eval_results['nights_total'] = (imagenet_score.item() * len(test_dataset_imagenet) +
                                    no_imagenet_score.item() * len(test_dataset_no_imagenet)) / total_length
    logging.info(f"NIGHTS (validation 2AFC): {str(eval_results['nights_val'])}")
    logging.info(f"NIGHTS (imagenet 2AFC): {str(eval_results['nights_imagenet'])}")
    logging.info(f"NIGHTS (no-imagenet 2AFC): {str(eval_results['nights_no_imagenet'])}")
    logging.info(f"NIGHTS (total 2AFC): {str(eval_results['nights_total'])}")
    return eval_results

def eval_bapps(model, preprocess, device, args):
    test_dataset_bapps = BAPPSDataset(root_dir=args.bapps_root, preprocess=preprocess)
    test_loader_bapps = DataLoader(test_dataset_bapps, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    bapps_score = score_bapps_dataset(model, test_loader_bapps, device)
    logging.info(f"BAPPS (total 2AFC): {str(bapps_score)}")
    return {"bapps_total": bapps_score}

def eval_things(model, preprocess, device, args):
    test_dataset_things = ThingsDataset(root_dir=args.things_root, txt_file=args.things_file, preprocess=preprocess)
    test_loader_things = DataLoader(test_dataset_things, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    things_score = score_things_dataset(model, test_loader_things, device)
    logging.info(f"THINGS (total 2AFC): {things_score}")
    return {"things_total": things_score}

def full_eval(eval_model, preprocess, device, args):
    results = {}
    if args.nights_root is not None:
        results['nights'] = eval_nights(eval_model, preprocess, device, args)
    if args.bapps_root is not None:
        results['bapps'] = bapps_results = eval_bapps(eval_model, preprocess, device, args)
    if args.things_root is not None:
        results['things'] = eval_things(eval_model, preprocess, device, args)
    return results
    
def run(args, device):
    logging.basicConfig(filename=os.path.join(args.eval_checkpoint, 'eval.log'), level=logging.INFO, force=True)
    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    os.makedirs(args.output, exist_ok=True)
    
    full_results = {}
    if args.eval_checkpoint is not None:
        eval_model, preprocess = load_dreamsim_model(args)
        full_results['ckpt'] = full_eval(eval_model, preprocess, device, args)
    if args.baseline_model is not None:
        baseline_model, baseline_preprocess = load_baseline_model(args)
        full_results['baseline'] = full_eval(baseline_model, baseline_preprocess, device, args)

    tag = args.tag + "_" if len(args.tag) > 0 else ""
    with open(os.path.join(args.output, f"{tag}eval_results.json"), "w") as f:
        json.dump(full_results, f)
    

if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args, device)
    
