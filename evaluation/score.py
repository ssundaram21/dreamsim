import torch
import os
from tqdm import tqdm
import logging
import numpy as np
import json
import torch.nn.functional as F

def score_nights_dataset(model, test_loader, device):
    logging.info("Evaluating NIGHTS dataset.")
    d0s = []
    d1s = []
    targets = []
    with torch.no_grad():
        for i, (img_ref, img_left, img_right, target, idx) in tqdm(enumerate(test_loader), total=len(test_loader)):
            img_ref, img_left, img_right, target = img_ref.to(device), img_left.to(device), \
                img_right.to(device), target.to(device)

            dist_0 = model(img_ref, img_left)
            dist_1 = model(img_ref, img_right)

            if len(dist_0.shape) < 1:
                dist_0 = dist_0.unsqueeze(0)
                dist_1 = dist_1.unsqueeze(0)
            dist_0 = dist_0.unsqueeze(1)
            dist_1 = dist_1.unsqueeze(1)
            target = target.unsqueeze(1)

            d0s.append(dist_0)
            d1s.append(dist_1)
            targets.append(target)

    d0s = torch.cat(d0s, dim=0)
    d1s = torch.cat(d1s, dim=0)
    targets = torch.cat(targets, dim=0)
    scores = (d0s < d1s) * (1.0 - targets) + (d1s < d0s) * targets + (d1s == d0s) * 0.5
    twoafc_score = torch.mean(scores, dim=0)
    print(f"2AFC score: {str(twoafc_score)}")
    return twoafc_score

def score_things_dataset(model, test_loader, device):
    logging.info("Evaluating Things dataset.")
    count = 0
    total = 0
    with torch.no_grad():
        for i, (img_1, img_2, img_3) in tqdm(enumerate(test_loader), total=len(test_loader)):
            img_1, img_2, img_3 = img_1.to(device), img_2.to(device), img_3.to(device)

            dist_1_2 = model(img_1, img_2)
            dist_1_3 = model(img_1, img_3)
            dist_2_3 = model(img_2, img_3)

            le_1_3 = torch.le(dist_1_2, dist_1_3)
            le_2_3 = torch.le(dist_1_2, dist_2_3)

            count += sum(torch.logical_and(le_1_3, le_2_3))
            total += len(torch.logical_and(le_1_3, le_2_3))
    count = count.detach().cpu().numpy()
    accs = count / total
    return accs

def score_bapps_dataset(model, test_loader, device):
    logging.info("Evaluating BAPPS dataset.")

    d0s = []
    d1s = []
    ps = []
    with torch.no_grad():
        for i, (im_ref, im_left, im_right, p) in tqdm(enumerate(test_loader), total=len(test_loader)):
            im_ref, im_left, im_right, p = im_ref.to(device), im_left.to(device), im_right.to(device), p.to(device)
            d0 = model(im_ref, im_left)
            d1 = model(im_ref, im_right)
            d0s.append(d0)
            d1s.append(d1)
            ps.append(p.squeeze())
    d0s = torch.cat(d0s, dim=0)
    d1s = torch.cat(d1s, dim=0)
    ps = torch.cat(ps, dim=0)
    scores = (d0s < d1s) * (1.0 - ps) + (d1s < d0s) * ps + (d1s == d0s) * 0.5
    final_score = torch.mean(scores, dim=0)
    return final_score
