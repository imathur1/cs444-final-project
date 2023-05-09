# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.


import argparse
import os
import pandas as pd
import sys
import torch
from torchvision import transforms
from tqdm import tqdm

from score import abs_rel, rsme, si_log, mae, calculate_mean_score, sq_rel
from utils import pfm_to_pil, load_image, visualize_depth_map
from lstsq import compute_scale_and_shift


SCORE_TYPES = {
    "SI Log": si_log,
    "MAE": mae,
    "RSME": rsme,
    "sq_rel": sq_rel,
    "abs_rel": abs_rel
}


def evaluate_depth(gt_path, pred_path, to_tensor):
    gt = load_image(gt_path, mode='depth')
    gt_tensor = to_tensor(gt)
    valid_mask = gt_tensor != 0.0
    mask_gt = gt_tensor == 0.0
    disp_tensor = 1 / gt_tensor
    gt_tensor[mask_gt] = float('nan')
    
    pred = pfm_to_pil(pred_path)
    pred_tensor = to_tensor(pred)
    
    pred_scale, pred_shift = compute_scale_and_shift(pred_tensor, disp_tensor, valid_mask)

    prediction_aligned = pred_scale.view(-1, 1, 1) * pred_tensor + pred_shift.view(-1, 1, 1)
    prediction_depth = 1.0 / prediction_aligned
    
    return prediction_depth, gt_tensor


def evaluate_disparity(gt_path, pred_path, to_tensor):
    # HERE, copy how they do it here in midas loss/run
    disp = load_image(gt_path, mode='depth')  # inverse depth/disparity uses same format as depth
    disp_tensor = to_tensor(disp)
    mask_disp = disp_tensor != 0.0
    
    pred = pfm_to_pil(pred_path)
    pred_tensor = to_tensor(pred)
    pred_scale, pred_shift = compute_scale_and_shift(pred_tensor, disp_tensor, mask_disp)

    prediction_aligned = pred_scale.view(-1, 1, 1) * pred_tensor + pred_shift.view(-1, 1, 1)
    disp_tensor[disp_tensor == 0.0] = float('nan')
    
    return prediction_aligned, disp_tensor


def write_mean_scores(scores):
    lines = []
    for score_name in SCORE_TYPES:
        values = [d[score_name] for n, d in scores.items()]
        mean_score = calculate_mean_score(values)
        lines.append(f"Mean {score_name}: {mean_score}")
    print(" || ".join(lines))
    

def save_viz(image, target, pred, sample_name, output_path):
    output_filename = os.path.join(output_path, sample_name)
    visualize_depth_map(image, target, pred, output_filename)


def get_args():
    parser = argparse.ArgumentParser(description="Calculate SILog and MAE for prediction of depth images")
    
    parser.add_argument("-gt", type=str, metavar="ground_truth", help="Path to directory containing the ground-truth")
    parser.add_argument("-pred", type=str, metavar="predictions", help="Path to directory containing the predictions")
    
    parser.add_argument("-o", type=str, metavar="output_path", help="Path to output directory")
    parser.add_argument("--inv_depth", action="store_true", help="If provided it will take the ground-truth as inverse depth images instead of depth")

    parser.add_argument("--viz", action="store_true", help="whether to save a visualization")
    parser.add_argument("--input_path", type=str, metavar="input_path", help="path to input for visualization", default=None)

    args = parser.parse_args()
    
    if args.viz and args.input_path is None:
        print("If --viz is specified --input_path must be specified too")
        sys.exit()
    
    # os.makedirs(args.o, exist_ok=True)
    # if args.viz:
    #     os.makedirs(os.path.join(args.o, "viz"), exist_ok=True)
    
    return args

def main():
    args = get_args()
    
    eval_func = evaluate_depth
    if args.inv_depth:
        eval_func = evaluate_disparity
    
    filenames = os.listdir(args.gt)
    if len(filenames) == 0:
        print(f"No files foung in GT folder")
        sys.exit()
    
    to_tensor = transforms.ToTensor()
    
    filenames.sort()
    scores = {}
    for fname in tqdm(filenames):
        scores[fname] = {}
        
        gt_path = os.path.join(args.gt, fname)
        pred_path = os.path.join(args.pred, fname.replace(".png", ".pfm"))  
        
        pred, gt = eval_func(gt_path, pred_path, to_tensor)
        
        for score_name, score in SCORE_TYPES.items():
            scores[fname][score_name] = score(pred, gt).item()
        
        if args.viz:
            ipt = load_image(os.path.join(args.input_path, fname), mode='RGB')
            viz_path = os.path.join(args.o, "viz")
            save_viz(ipt, gt.squeeze().numpy(), pred.squeeze().numpy(), fname.replace(".png", ""), viz_path)
    
    write_mean_scores(scores)
    
    # save individual scores 
    filename = os.path.join(args.o, "scores.csv")
    df = pd.DataFrame.from_dict(scores, orient="index")
    df.to_csv(filename)


if __name__ == "__main__":
    main()