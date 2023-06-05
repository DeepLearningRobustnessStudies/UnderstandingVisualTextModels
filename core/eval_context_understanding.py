import copy

import numpy as np
import datetime
from pathlib import Path
import logging
import argparse
import json
from tqdm import tqdm
from PIL import Image
import torch
import random
import pandas as pd
import pdb
import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
import torchvision.transforms as transforms
from torchvision.transforms import Compose

from transformers import AutoProcessor
import sys, os, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
from UnderstandingVisualTextModels.models.CLIP import clip


class BackgroundCuesDataset(Dataset):
    def __init__(self, annot_dir, image_root_original, image_root_modified, image_root_patch):
        """

        :param annot_dir:
        :param image_root_original: Where the original images are stored, should be `val2014`
        :param image_root_modified: Where modified images are based on segmentation/bbox and different fillers
        """
        self.image_root_original = image_root_original
        self.image_root_modified = image_root_modified
        self.image_root_patch = image_root_patch

        pth = os.path.join(annot_dir, 'background_context_dataset.json')
        self.annotations = json.load(open(pth, 'r'))

        # Filter for ones we were able to get patches for
        patched = os.listdir(image_root_patch)
        orig_len = len(self)
        self.annotations = [x for x in self.annotations if x['image'] in patched]
        print(f"Number of annotations kept {len(self)}/{orig_len}")

        # Get all classes for multilabel recognition, removing apple because too much overlap
        self.coco_classes = pd.read_csv(os.path.join(annot_dir, 'prompt_classes.csv'))
        self.class_prompts = self.coco_classes['prompt'].values
        self.coco_classes = np.array(self.coco_classes['class'].values)

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def load_image(image_root, image_id):
        # Load image that represents the actual relationship
        pth = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
        f = open(os.path.join(image_root, pth), 'rb')
        image = Image.open(f).convert("RGB")
        image.load()
        f.close()
        return image

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        image_id = annotation['image_id']
        image_original = self.load_image(self.image_root_original, image_id)
        image_modified = self.load_image(self.image_root_modified, image_id)
        image_patch = self.load_image(self.image_root_patch, image_id)

        labels = torch.zeros(len(self.coco_classes))
        objects = list(annotation['annotations'].keys())

        for obj in objects:
            labels[np.where(self.coco_classes == obj)[0][0]] = 1.0

        return image_original, image_modified, image_patch, labels

    def collate_fn(self, batch):
        images = [x[0] for x in batch]
        images_modified = [x[1] for x in batch]
        images_patch = [x[2] for x in batch]
        labels = [x[3] for x in batch]
        return images, images_modified, images_patch, labels


def average_precision(target, output):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def make_comparisons(gt_labels, logits, results):

    gt_logits = logits[0].softmax(dim=0).cpu().numpy()
    gt_ap = average_precision(gt_labels, gt_logits)

    patch_logits = logits[1].softmax(dim=0).cpu().numpy()
    patch_ap = average_precision(gt_labels, patch_logits)

    mod_logits = logits[2].softmax(dim=0).cpu().numpy()
    mod_ap = average_precision(gt_labels, mod_logits)

    softmax_mods = mod_logits[np.argwhere(gt_labels > 0)].transpose()[0]
    softmax_gt = gt_logits[np.argwhere(gt_labels > 0)].transpose()[0]
    softmax_patch = patch_logits[np.argwhere(gt_labels > 0)].transpose()[0]

    results['gt_ap'].append(float(gt_ap))
    results['mod_ap'].append(float(mod_ap))
    results['patch_ap'].append(float(patch_ap))

    results['change_gt_mod_ap'].append(float(gt_ap - mod_ap))
    results['change_gt_patch_ap'].append(float(gt_ap - patch_ap))
    results['change_patch_mod_ap'].append(float(patch_ap - mod_ap))

    results['relative_robustness_gt_mod_ap'].append(float(1 - (gt_ap - mod_ap) / gt_ap))
    results['relative_robustness_gt_patch_ap'].append(float(1 - (gt_ap - patch_ap) / gt_ap))
    results['relative_robustness_patch_mod_ap'].append(float(1 - (patch_ap - mod_ap) / patch_ap))

    results['softmax_mods'].append(softmax_mods.tolist())
    results['softmax_gt'].append(softmax_gt.tolist())
    results['softmax_patch'].append(softmax_patch.tolist())

    results['change_gt_mod_conf'].append(float(np.mean(softmax_gt - softmax_mods)))
    results['change_patch_mod_conf'].append(float(np.mean(softmax_patch - softmax_mods)))
    results['change_gt_patch_conf'].append(float(np.mean(softmax_gt - softmax_patch)))
    return results

def eval_clip(model_name, class_prompts, pbar, results):
    """

    :param model_name:
    :param class_prompts:
    :param pbar:
    :param results:
    :return:
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model_type = model_name.split('_')[-1]
    logger.info(f"Building {clip_model_type} based CLIP model...")

    # model_name = clip_model_type.replace("/", "")
    model, preprocess = clip.load(clip_model_type, device)
    model.eval()
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)

    # Get text prompts
    with torch.no_grad():
        text_features = model.encode_text(class_prompts)
        if isinstance(text_features, tuple):
            text_features = text_features[0]
        text_features = normalize(text_features)

    logit_scale = model.logit_scale.exp().detach()

    for image_orig, image_mod, image_patch, labels in pbar:
        with torch.no_grad():
            images = torch.stack([preprocess(image) for image in image_orig + image_patch + image_mod]).to(device)

            # Get features
            visual_features = model.encode_image(images)
            if isinstance(visual_features, tuple):
                visual_features = visual_features[0]

        visual_features = normalize(visual_features)

        gt_labels = np.stack(labels)[0]
        logits = get_logits(logit_scale, visual_features, text_features).t()
        results = make_comparisons(gt_labels, logits, results)

    return model_name, results


def eval_flava(class_prompts, pbar, results):
    """
    transformers version==4.26.0
    https://huggingface.co/spaces/flava/zero-shot-image-classification/blob/main/app.py
    :param class_prompts:
    :param pbar:
    :param results:
    :return:
    """
    from transformers import FlavaProcessor, FlavaForPreTraining
    from transformers import FlavaModel, BertTokenizer, FlavaFeatureExtractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device {device}")
    # model = FlavaForPreTraining.from_pretrained("facebook/flava-full").to(device)
    # preprocess = FlavaProcessor.from_pretrained("facebook/flava-full")

    model = FlavaModel.from_pretrained("facebook/flava-full").to(device)
    model.eval()
    fe = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
    tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")

    text_features = list()
    with torch.no_grad():
        for prompt in class_prompts.tolist():
            text_inputs = tokenizer([prompt], padding="max_length", return_tensors="pt").to(device)
            text_embeddings = model.get_text_features(**text_inputs)[:, 0, :]
            text_features.append(text_embeddings.cpu())

    text_features = torch.cat(text_features).cuda()

    for image_orig, image_mod, image_patch, labels in pbar:
        with torch.no_grad():
            image_input = fe(image_orig + image_patch + image_mod, return_tensors="pt").to(device)
            image_embeddings = model.get_image_features(**image_input)[:, 0, :]
            logits = torch.nn.functional.softmax((text_features @ image_embeddings.T).squeeze(0), dim=0).t()

        # Ground Truth comparison
        gt_labels = np.stack(labels)[0]
        results = make_comparisons(gt_labels, logits, results)

    return results


def eval_bridgetower(class_prompts, pbar, results):
    """
    This is incredibly slow, very very depressing.
    Trying to increase speed, requires > 48GB of capacity which is ridiculuos
    https://huggingface.co/docs/transformers/main/en/model_doc/bridgetower#overview
    :param class_prompts:
    :param pbar:
    :param results:
    :return:
    """
    local=False
    from transformers import BridgeTowerForImageAndTextRetrieval, BridgeTowerProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "BridgeTower/bridgetower-large-itm-mlm"
    preprocess = BridgeTowerProcessor.from_pretrained(model_id)
    model = BridgeTowerForImageAndTextRetrieval.from_pretrained(model_id).to(device)
    model.eval()

    num_classes = len(class_prompts)

    text_inputs = list()
    for image_orig, image_mod, image_patch, labels in pbar:
        with torch.no_grad():
            if len(text_inputs) < 1:
                text_inputs = preprocess(images=image_orig,
                                         text=class_prompts.tolist(),
                                         return_tensors="pt",
                                         padding="max_length",
                                         truncation=False,
                                         max_length=40).to(device)


            inputs = preprocess(images=image_orig + image_mod + image_patch,
                                text=["test"],
                                return_tensors="pt",
                                padding="max_length",
                                truncation=False,
                                max_length=40).to(device)
            if not local:
                outputs = list()
                for i in range(3):
                    tmp = copy.deepcopy(text_inputs)
                    tmp['pixel_values'] = inputs['pixel_values'][i].unsqueeze(0).repeat(num_classes, 1, 1, 1)
                    tmp['pixel_mask'] = inputs['pixel_values'][i].unsqueeze(0).repeat(num_classes, 1, 1, 1)
                    # with torch.autocast('cuda'):
                    outputs.append(model(**tmp.to(device)).logits[:, 1].cpu())
            else:
                text_inputs['pixel_values'] = inputs['pixel_values']
                text_inputs['pixel_mask'] = inputs['pixel_mask']

                outputs = list()
                for i in range(len(class_prompts)):
                    tmp = copy.deepcopy(text_inputs)
                    tmp['input_ids'] = tmp['input_ids'][i].unsqueeze(0).repeat(3, 1)
                    tmp['attention_mask'] = tmp['attention_mask'][i].unsqueeze(0).repeat(3, 1)
                    # tmp['token_type_ids'] = tmp['token_type_ids'][i].unsqueeze(0).repeat(3, 1)
                    pdb.set_trace()
                    outputs.append(model(**tmp.to(device)).logits[:, 1].cpu())

        logits = torch.stack(outputs).float() # .t()
        if local:
            logits = logits.t()
        # Ground Truth comparison
        gt_labels = np.stack(labels)[0]

        results = make_comparisons(gt_labels, logits, results)
    return results


def eval_vilt(class_prompts, pbar, results):
    """
    transformers version==4.26.0
    # https://arxiv.org/pdf/2102.03334.pdf
    # https://huggingface.co/spaces/MikailDuzenli/vilt_demo

    :param class_prompts:
    :param pbar:
    :param results:
    :return:
    """

    from transformers import ViltProcessor, ViltForImageAndTextRetrieval
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")

    model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")
    model.eval()
    model.to(device)

    text_inputs = list()
    for image_orig, image_mod, image_patch, labels in pbar:
        with torch.no_grad():
            if len(text_inputs) < 1:
                text_inputs = preprocess(images=image_orig,
                                    text=class_prompts.tolist(),
                                    return_tensors="pt",
                                    padding="max_length",
                                    truncation=False,
                                    max_length=40).to(device)

            inputs = preprocess(images=image_orig+image_mod+image_patch,
                                text=["test"],
                                    return_tensors="pt",
                                    padding="max_length",
                                    truncation=False,
                                    max_length=40).to(device)
            text_inputs['pixel_values'] = inputs['pixel_values']
            text_inputs['pixel_mask'] = inputs['pixel_mask']

            outputs = list()
            for i in range(len(class_prompts)):
                tmp = copy.deepcopy(text_inputs)
                tmp['input_ids'] = tmp['input_ids'][i].unsqueeze(0).repeat(3, 1)
                tmp['attention_mask'] = tmp['attention_mask'][i].unsqueeze(0).repeat(3, 1)
                tmp['token_type_ids'] = tmp['token_type_ids'][i].unsqueeze(0).repeat(3, 1)

                outputs.append(model(**tmp.to(device)).logits)

        logits = torch.stack(outputs)[:, :, 0].t()

        # Ground Truth comparison
        gt_labels = np.stack(labels)[0]
        results = make_comparisons(gt_labels, logits, results)

    return results


def eval(model_name,
         annot_dir,
         image_root_original,
         image_root_modified,
         image_root_patch,
         fill_type,
         num_workers,
         batch_size,
         save_dir):
    """
    This will compare drop in performance between images with the original background and images with the
        background removed and filled with either "black", "gray", "noise" or a randomly selected landscape
    :param model_name:
    :param annot_dir:
    :param image_root_original:
    :param image_root_modified:
    :param fill_type:
    :param num_workers:
    :param batch_size:
    :param save_dir:
    :return:
    """
    logger.info("Building dataset...")
    logger.info(f"Running evaluation for when background is removed...")

    dataset = BackgroundCuesDataset(annot_dir, image_root_original, image_root_modified, image_root_patch)
    class_prompts = dataset.class_prompts
    dataloader = DataLoader(dataset, num_workers=num_workers, drop_last=False, batch_size=batch_size,
                            collate_fn=dataset.collate_fn)

    logger.info(f"Done building dataset with {len(dataset)} total samples and {len(dataloader)} batches.")
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Running for {model_name} removing background")

    # Create our dictionary of results.
    results = dict()
    columns = ['gt_ap',
               'mod_ap',
               'change_gt_mod_ap',
               'change_gt_patch_ap',
               'change_patch_mod_ap',
               'patch_ap',
               'relative_robustness_gt_mod_ap',
               'relative_robustness_gt_patch_ap',
               'relative_robustness_patch_mod_ap',
               'softmax_mods',
               'softmax_gt',
               'softmax_patch',
               'change_gt_mod_conf',
               'change_patch_mod_conf',
               'change_gt_patch_conf']

    for col in columns:
        results[col] = list()

    if 'clip' in model_name:
        model_name, results = eval_clip(model_name, class_prompts, pbar, results)
    elif model_name == 'flava':
        results = eval_flava(class_prompts, pbar, results)
    elif model_name == 'vilt':
        results = eval_vilt(class_prompts, pbar, results)
    elif model_name == 'bridgetower':
        results = eval_bridgetower(class_prompts, pbar, results)
    else:
        raise NotImplementedError

    save_dir = os.path.join(save_dir,'results', model_name.replace('/', '').replace('@', ''))
    if not os.path.exists(save_dir):
        Path(save_dir).mkdir(parents=True)

    with open(os.path.join(save_dir, f'eval_context_background_{fill_type}_understanding.json'),
              'w') as f:
        json.dump(results, f)

    logger.info("-----------------------")
    logger.info(f"Softmax for {fill_type}:")
    agg = {k:v for k, v in results.items() if 'softmax' not in k}
    for comparison, scores in agg.items():
        if len(scores) > 0:
            agg[comparison] = round(np.mean(np.array(scores)), 4)
            logger.info(f"Results for {comparison}: {agg[comparison]}")

    with open(os.path.join(save_dir,
                           f'eval_aggregated_context_background_{fill_type}_understanding.json'), 'w') as f:
        json.dump(agg, f)


@torch.jit.script
def normalize(x):
    return x / x.norm(dim=-1, keepdim=True, p=2)


@torch.jit.script
def get_logits(logit_scale, imf_i, text_f):
    # return logit_scale * imf_i @ text_f.t()
    return logit_scale * text_f @ imf_i.t()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs  a COCO based evaluation on CLIP.')
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='/home/mschiappa/SRI/UnderstandingVisualTextModels/results')
    # Dataset Specific
    parser.add_argument('--annot_dir', type=str,
                        default='/home/mschiappa/SRI/UnderstandingVisualTextModels/new_datasets/context')
    parser.add_argument('--root_image_orig_dir', default='/media/mschiappa/Elements/coco/val2014', type=str)
    parser.add_argument('--root_image_patch_dir', default='/media/mschiappa/Elements/coco/val2014_random_patch_dataset',
                        type=str)
    parser.add_argument('--root_image_mod_dir', default='/media/mschiappa/Elements/coco/val2014_background_removed_dataset', type=str)
    parser.add_argument('--fill_type', type=str, default='noise')
    # parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # Build our save directory for results, must pass the top-level directory results are stored in
    save_dir = os.path.join(args.save_dir, 'context_text_eval', args.model.replace('/', '').replace('@', ''))
    save_dir = Path(save_dir)
    if not os.path.exists(save_dir):
        Path.mkdir(save_dir, parents=True)

    # Logger initialization
    log_path = os.path.join(save_dir, f'log.txt')
    if os.path.isfile(log_path):
        log_path = os.path.join(save_dir, f'log_{datetime.datetime.now().strftime("%d-%m-%H%M%S")}.txt')

    # Create a Logger Object - Which listens to everything
    logger = logging.getLogger(os.path.basename(__file__))
    logger.setLevel(logging.DEBUG)

    # Register the Console as a handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Log format includes date and time
    formatter = logging.Formatter('%(asctime)s %(levelname)-5s %(message)s')
    ch.setFormatter(formatter)

    # If want to print output to screen
    logger.addHandler(ch)

    # Create a File Handler to listen to everything
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)

    # Log format includes date and time
    fh.setFormatter(formatter)

    # Register it as a listener
    logger.addHandler(fh)
    print('-------------------------------')
    print(f"Run arguments:")
    for k in args.__dict__:
        print(f'{k}: {args.__dict__[k]}')
    print('-------------------------------')
    assert args.fill_type in ['noise', 'black', 'gray', 'scene']
    root_image_mod_dir = os.path.join(args.root_image_mod_dir, 'segmentation_mask', args.fill_type)
    root_image_patch_dir = os.path.join(args.root_image_patch_dir, args.fill_type)

    eval(args.model, args.annot_dir, args.root_image_orig_dir, root_image_mod_dir, root_image_patch_dir, args.fill_type,
         args.num_workers, 1, save_dir)