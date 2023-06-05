import numpy as np
import datetime
from pathlib import Path
import logging
import argparse
import json
from tqdm import tqdm
from PIL import Image
import warnings
import torch
import copy
import pandas as pd
from collections import Counter
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


class OccurenceDataset(Dataset):
    def __init__(self, annot_dir, image_root_original, image_root_modified, image_root_patch):
        """

        :param annot_dir:
        :param image_root_original: Where the original images are stored, should be `val2014`
        :param image_root_modified: Where modified images are based on segmentation/bbox and different fillers
        """
        self.image_root_original = image_root_original
        self.image_root_modified = image_root_modified
        self.image_root_patch = image_root_patch
        pth = os.path.join(annot_dir, 'object_context_dataset.json')
        self.annotations = json.load(open(pth, 'r'))

        # Filter for ones we were able to get patches for
        patched = os.listdir(image_root_patch)
        patched = [int(p.split('.')[0].split('_')[-1]) for p in patched]
        orig_len = len(self)
        self.annotations = [x for x in self.annotations if x['image_id'] in patched]
        # print(f"Number of annotations kept {len(self)}/{orig_len}")

        # Filter for background
        original = os.listdir(image_root_original)
        original = [int(p.split('.')[0].split('_')[-1]) for p in original]
        self.annotations = [x for x in self.annotations if x['image_id'] in original and x['object'] != 'apple']

        # Filter for ones that actually exist...
        mod = os.listdir(image_root_modified)
        self.annotations = [x for x in self.annotations if f"{x['object']}_COCO_val2014_{str(x['image_id']).zfill(12)}.jpg" in mod]

        print(f"Number of annotations kept {len(self)}/{orig_len}")

        # Get all classes for multilabel recognition, removing apple because too much overlap
        self.coco_classes = pd.read_csv(os.path.join(annot_dir, 'prompt_classes.csv'))
        self.coco_classes = self.coco_classes[self.coco_classes['class'] != 'apple']
        self.class_prompts = self.coco_classes['prompt'].values
        self.coco_classes = np.array(self.coco_classes['class'].values)

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def load_image(image_root, image_id, object=None):
        # Load image that represents the actual relationship
        if object is not None:
            pth = f"{object}_COCO_val2014_{str(image_id).zfill(12)}.jpg"
        else:
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
        image_modified = self.load_image(self.image_root_modified, image_id, annotation['object'])
        image_patch = self.load_image(self.image_root_patch, image_id)

        # Single object label
        try:
            label = np.where(self.coco_classes == annotation['object'])[0][0]
        except:
            pdb.set_trace()

        # All object labels
        labels = torch.zeros(len(self.coco_classes))
        objects = [annotation['object']] + annotation['other_objects']

        for obj in objects:
            if obj != 'apple':
                labels[np.where(self.coco_classes == obj)[0][0]] = 1.0

        return image_original, image_modified, image_patch, label, labels, annotation['object'], annotation['other_objects']

    def collate_fn(self, batch):
        images = [x[0] for x in batch]
        images_modified = [x[1] for x in batch]
        images_patched = [x[2] for x in batch]
        label = [x[3] for x in batch]
        labels = [x[4] for x in batch]
        single_object = [x[5] for x in batch]
        other_objects = [x[6] for x in batch]
        return images, images_modified,images_patched, label, labels, single_object, other_objects


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


def make_comparisons(gt_labels, label, single_object, other_objects, logits, results,
                     no_others_logits, new_label):

    gt_no_others_logits = no_others_logits[0].softmax(dim=0).cpu().numpy()
    patch_no_others_logits = no_others_logits[1].softmax(dim=0).cpu().numpy()
    mod_no_others_logits = no_others_logits[2].softmax(dim=0).cpu().numpy()

    # single object softmax, softmax when the other objects in image are not compared to
    results['gt_single_object_softmax'].append(float(gt_no_others_logits[new_label]))
    results['patch_single_object_softmax'].append(float(patch_no_others_logits[new_label]))
    results['mod_single_object_softmax'].append(float(mod_no_others_logits[new_label]))

    # Single object accuracy
    results['gt_single_object_acc'].append(float(np.argmax(gt_no_others_logits) == new_label))
    results['patch_single_object_acc'].append(float(np.argmax(patch_no_others_logits) == new_label))
    results['mod_single_object_acc'].append(float(np.argmax(mod_no_others_logits) == new_label))

    # Change in confidence
    results['gt_mod_change_conf'].append(float(gt_no_others_logits[new_label] - mod_no_others_logits[new_label]))
    results['gt_patch_change_conf'].append(float(gt_no_others_logits[new_label] - patch_no_others_logits[new_label]))
    results['patch_mod_change_conf'].append(float(patch_no_others_logits[new_label] - mod_no_others_logits[new_label]))

    # Relative Robustness
    results['rel_robustness_gt_mod_change_conf'].append(
        float(1 - (gt_no_others_logits[new_label] - mod_no_others_logits[new_label]) / gt_no_others_logits[new_label]))
    results['rel_robustness_gt_patch_change_conf'].append(
        float(1 - (gt_no_others_logits[new_label] - patch_no_others_logits[new_label]) / gt_no_others_logits[new_label]))
    results['rel_robustness_patch_mod_change_conf'].append(
        float(1 - (patch_no_others_logits[new_label] - mod_no_others_logits[new_label]) / patch_no_others_logits[new_label]))

    # Absolute Robustness
    results['abs_robustness_gt_mod_change_conf'].append(
        float(1 - (gt_no_others_logits[new_label] - mod_no_others_logits[new_label]) / 1.0))
    results['abs_robustness_gt_patch_change_conf'].append(
        float(
            1 - (gt_no_others_logits[new_label] - patch_no_others_logits[new_label]) / 1.0))
    results['abs_robustness_patch_mod_change_conf'].append(
        float(1 - (patch_no_others_logits[new_label] - mod_no_others_logits[new_label]) / 1.0))

    results['single_object'].append(single_object[0])
    results['other_objects'].append(other_objects[0])

    return results


def eval_clip(model_name, class_prompts, pbar, results):
    """
    TODO: remove other objects present in image?
    :param model_name:
    :param class_prompts:
    :param pbar:
    :param results:
    :return:
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_type = model_name.split('_')[-1]
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

    for image_orig, image_mod, image_patch, label, labels, single_object, other_objects in pbar:
        tmp = {i: p for i, p in enumerate(class_prompts) if p.replace('A photo of a ', '').replace('.', '') not in other_objects[0]}
        tmp_text_features = text_features[list(tmp.keys())]
        new_label = [i for i, p in enumerate(tmp.values()) if p.replace('A photo of a ', '').replace('.', '') == single_object[0]]

        with torch.no_grad():
            images = torch.stack([preprocess(image) for image in image_orig + image_patch + image_mod ]).to(device)

            # Get features
            visual_features = model.encode_image(images)
            if isinstance(visual_features, tuple):
                visual_features = visual_features[0]

        visual_features = normalize(visual_features)

        gt_labels = np.stack(labels)[0]

        all_logits = get_logits(logit_scale, visual_features, text_features).t()
        no_others_logits = get_logits(logit_scale, visual_features, tmp_text_features).t()

        results = make_comparisons(gt_labels, label, single_object, other_objects, all_logits, results,
                                   no_others_logits, new_label)

    return model_name, results


def eval_flava(class_prompts, pbar, results):
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

    for image_orig, image_mod, image_patch, label, labels, single_object, other_objects in pbar:
        tmp = {i: p for i, p in enumerate(class_prompts) if
               p.replace('A photo of a ', '').replace('.', '') not in other_objects[0]}
        tmp_text_features = text_features[list(tmp.keys())]
        new_label = [i for i, p in enumerate(tmp.values()) if
                     p.replace('A photo of a ', '').replace('.', '') == single_object[0]]
        with torch.no_grad():
            image_input = fe(image_orig + image_patch + image_mod, return_tensors="pt").to(device)
            image_embeddings = model.get_image_features(**image_input)[:, 0, :]
            # logits = torch.nn.functional.softmax((text_features @ image_embeddings.T).squeeze(0), dim=0).t()
            all_logits = (text_features @ image_embeddings.T).squeeze(0).t()
            no_others_logits = (tmp_text_features @ image_embeddings.T).squeeze(0).t()

        gt_labels = np.stack(labels)[0]
        results = make_comparisons(gt_labels, label, single_object, other_objects, all_logits, results,
                                   no_others_logits, new_label)

    return results


def eval_vilt(class_prompts, pbar, results):
    # https://arxiv.org/pdf/2102.03334.pdf
    from transformers import ViltProcessor, ViltForImageAndTextRetrieval
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
    model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")
    model.to(device)
    text_inputs = list()
    for image_orig, image_mod, image_patch, label, labels, single_object, other_objects in pbar:
        tmp = {i: p for i, p in enumerate(class_prompts) if
               p.replace('A photo of a ', '').replace('.', '') not in other_objects[0]}
        no_others_class_prompts = list(tmp.values())
        new_label = [i for i, p in enumerate(tmp.values()) if
                     p.replace('A photo of a ', '').replace('.', '') == single_object[0]]

        with torch.no_grad():
            if len(text_inputs) < 1:
                text_inputs = preprocess(images=image_orig,
                                         text=class_prompts.tolist(),
                                         return_tensors="pt",
                                         padding="max_length",
                                         truncation=False,
                                         max_length=40).to(device)
            no_others_text_inputs = preprocess(images=image_orig + image_mod + image_patch,
                                         text=no_others_class_prompts,
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
            text_inputs['pixel_values'] = inputs['pixel_values']
            text_inputs['pixel_mask'] = inputs['pixel_mask']

            outputs = list()
            no_others_outputs = list()
            for i in range(len(class_prompts)):
                tmp = copy.deepcopy(text_inputs)
                tmp['input_ids'] = tmp['input_ids'][i].unsqueeze(0).repeat(3, 1)
                tmp['attention_mask'] = tmp['attention_mask'][i].unsqueeze(0).repeat(3, 1)
                tmp['token_type_ids'] = tmp['token_type_ids'][i].unsqueeze(0).repeat(3, 1)

                outputs.append(model(**tmp.to(device)).logits)

                if i < len(no_others_class_prompts):
                    tmp = copy.deepcopy(no_others_text_inputs)
                    tmp['input_ids'] = tmp['input_ids'][i].unsqueeze(0).repeat(3, 1)
                    tmp['attention_mask'] = tmp['attention_mask'][i].unsqueeze(0).repeat(3, 1)
                    tmp['token_type_ids'] = tmp['token_type_ids'][i].unsqueeze(0).repeat(3, 1)
                    no_others_outputs.append(model(**tmp.to(device)).logits)


        all_logits = torch.stack(outputs)[:, :, 0].t()
        no_others_logits = torch.stack(no_others_outputs)[:, :, 0].t()
        gt_labels = np.stack(labels)[0]
        results = make_comparisons(gt_labels, label, single_object, other_objects, all_logits, results,
                                   no_others_logits, new_label)
    return results

def eval_bridgetower(class_prompts, pbar, results):
    # https://arxiv.org/pdf/2102.03334.pdf
    from transformers import BridgeTowerForImageAndTextRetrieval, BridgeTowerProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "BridgeTower/bridgetower-large-itm-mlm"
    preprocess = BridgeTowerProcessor.from_pretrained(model_id)
    model = BridgeTowerForImageAndTextRetrieval.from_pretrained(model_id).to(device)
    model.eval()
    text_inputs = list()
    for image_orig, image_mod, image_patch, label, labels, single_object, other_objects in pbar:
        tmp = {i: p for i, p in enumerate(class_prompts) if
               p.replace('A photo of a ', '').replace('.', '') not in other_objects[0]}
        no_others_class_prompts = list(tmp.values())
        new_label = [i for i, p in enumerate(tmp.values()) if
                     p.replace('A photo of a ', '').replace('.', '') == single_object[0]]
        with torch.no_grad():
            # this is so we can preprocess the prompts just one time
            # if len(text_inputs) < 1:
            #     text_inputs = preprocess(images=image_orig,
            #                              text=class_prompts.tolist(),
            #                              return_tensors="pt",
            #                              padding="max_length",
            #                              truncation=True,
            #                              max_length=40).to(device)

            # we have to preprocess every time because the objects removed changes
            no_others_text_inputs = preprocess(images=image_orig + image_mod + image_patch,
                                               text=no_others_class_prompts,
                                               return_tensors="pt",
                                               padding="max_length",
                                               truncation=True,
                                               max_length=40).to(device)
            num_classes = len(no_others_class_prompts)
            no_others_outputs = list()
            for i in range(3):
                tmp = copy.deepcopy(no_others_text_inputs)
                tmp['pixel_values'] = no_others_text_inputs['pixel_values'][i].unsqueeze(0).repeat(num_classes, 1, 1, 1)
                tmp['pixel_mask'] = no_others_text_inputs['pixel_values'][i].unsqueeze(0).repeat(num_classes, 1, 1, 1)

                no_others_outputs.append(model(**tmp.to(device)).logits[:, 1].cpu())
            # # This is for the images for the 80 prompts
            # inputs = preprocess(images=image_orig + image_mod + image_patch,
            #                     text=["test"],
            #                     return_tensors="pt",
            #                     padding="max_length",
            #                     truncation=False,
            #                     max_length=40).to(device)
            # text_inputs['pixel_values'] = inputs['pixel_values']
            # text_inputs['pixel_mask'] = inputs['pixel_mask']

            # outputs = list()
            # no_others_outputs = list()
            # for i in range(len(no_others_class_prompts)):
            #     # tmp = copy.deepcopy(text_inputs)
            #     # tmp['input_ids'] = tmp['input_ids'][i].unsqueeze(0).repeat(3, 1)
            #     # tmp['attention_mask'] = tmp['attention_mask'][i].unsqueeze(0).repeat(3, 1)
            #     # # tmp['token_type_ids'] = tmp['token_type_ids'][i].unsqueeze(0).repeat(3, 1)
            #     #
            #     # outputs.append(model(**tmp.to(device)).logits)
            #
            #     tmp = copy.deepcopy(no_others_text_inputs)
            #     tmp['input_ids'] = tmp['input_ids'][i].unsqueeze(0).repeat(3, 1)
            #     tmp['attention_mask'] = tmp['attention_mask'][i].unsqueeze(0).repeat(3, 1)
            #     # tmp['token_type_ids'] = tmp['token_type_ids'][i].unsqueeze(0).repeat(3, 1)
            #     with torch.autocast('cuda'):
            #         no_others_outputs.append(model(**tmp.to(device)).logits[:, 1].cpu())

        # all_logits = torch.stack(outputs)[:, :, 0].t()
        no_others_logits = torch.stack(no_others_outputs)#.t() # [:, :, 0].t()
        gt_labels = np.stack(labels)[0]
        results = make_comparisons(gt_labels, label, single_object, other_objects, None, results,
                                   no_others_logits, new_label)
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
    This will evaluate whether object detections are as confident when other objects aren't present
    Comparisons:
        * Original Image with background removed with all objects vs. Image with just the one object and background removed
        * Patched image with all objects vs. Image with just one object and background removed

    All segmentation masks.

    :param model:
    :param model_name:
    :param annot_dir:
    :param root_image_dir:
    :param num_workers:
    :param batch_size:
    :return:
    """
    logger.info("Building dataset...")
    logger.info(f"Running evaluation for object co-occurrence...")
    dataset = OccurenceDataset(annot_dir, image_root_original, image_root_modified, image_root_patch)
    class_prompts = dataset.class_prompts
    dataloader = DataLoader(dataset, num_workers=num_workers, drop_last=False, batch_size=batch_size,
                            collate_fn=dataset.collate_fn)
    logger.info(f"Done building dataset with {len(dataset)} total samples and {len(dataloader)} batches.")
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Running for {model_name}")

    # Create our dictionary of results.
    results = dict()
    columns = ['all_objects_ap',

               'gt_single_object_acc',
               'patch_single_object_acc',
               'mod_single_object_acc',

               'gt_single_object_softmax',
               'patch_single_object_softmax',
               'mod_single_object_softmax',

               'gt_all_objects_softmax',
               'patch_all_objects_softmax',
               'mod_all_objects_softmax',

               'gt_mod_change_conf',
               'gt_patch_change_conf',
               'patch_mod_change_conf',

               'rel_robustness_gt_mod_change_conf',
               'rel_robustness_gt_patch_change_conf',
               'rel_robustness_patch_mod_change_conf',

               'abs_robustness_gt_mod_change_conf',
               'abs_robustness_gt_patch_change_conf',
               'abs_robustness_patch_mod_change_conf',

               'single_object',
               'other_objects']

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

    save_dir = os.path.join(save_dir, 'results', model_name.replace('/', '').replace('@', ''))
    if not os.path.exists(save_dir):
        Path(save_dir).mkdir(parents=True)

    with open(os.path.join(save_dir, f'eval_cooccurence_{fill_type}_understanding.json'),
              'w') as f:
        json.dump(results, f)

    logger.info("-----------------------")

    agg = {k: v for k, v in results.items() if 'softmax' not in k}
    agg = {k: v for k, v in agg.items() if k not in ["single_object", "other_objects"]}

    for comparison, scores in agg.items():
        if len(scores) > 0:
            agg[comparison] = round(np.mean(np.array(scores)), 4)
            logger.info(f"Results for {comparison}: {agg[comparison]}")
    logger.info("-----------------------")

    with open(os.path.join(save_dir,
                           f'eval_aggregated_cooccurence_{fill_type}_understanding.json'), 'w') as f:
        json.dump(agg, f)

    object_means = dict()
    for obj, mod in zip(results['single_object'], results['mod_single_object_acc']):
        if obj not in object_means:
            object_means[obj] = list()
        object_means[obj].append(mod)
    object_means = {k: np.mean(v) for k, v in object_means.items()}
    logger.info("Mean accuracy on single object no background or other objects:")
    for k, v in object_means.items():
        logger.info(f"{k}: {v}")

    object_robustness = dict()
    object_difference = dict()

    for obj, patch, mod in zip(results['single_object'], results['patch_single_object_softmax'], results['mod_single_object_softmax']):
        if obj not in object_robustness:
            object_robustness[obj] = list()
            object_difference[obj] = list()
        object_robustness[obj].append(1 - (patch - mod) / patch)
        object_difference[obj].append(patch - mod)
    object_robustness = {k: np.mean(v) for k, v in object_robustness.items()}
    object_difference = {k: np.mean(v) for k, v in object_difference.items()}
    logger.info("\nMean robustness based on change in normalized logit for single object versus. all objects")
    for k, v in object_robustness.items():
        logger.info(f"{k}: {v}")

    # Want to look at the change in other object logits
    other_objects_diff = dict()
    other_objects_mod = dict()
    other_objects_patch = dict()
    for objs, patch, mod in zip(results['other_objects'], results['gt_all_objects_softmax'], results['mod_all_objects_softmax']):
        for obj in objs:
            if obj == 'apple':
                continue
            label = np.where(dataset.coco_classes == obj)[0][0]
            patch_logit = patch[label]
            mod_logit = mod[label]
            difference = patch_logit-mod_logit

            if obj not in other_objects_diff:
                other_objects_diff[obj] = list()
                other_objects_mod[obj] = list()
                other_objects_patch[obj] = list()
            other_objects_diff[obj].append(difference)
            other_objects_mod[obj].append(mod_logit)
            other_objects_patch[obj].append(patch_logit)

    other_objects_diff_agg = {k: np.mean(v) for k, v in other_objects_diff.items()}
    logger.info("\nMean drop in normalized logit for other objects when present versus when not.")
    for k, v in other_objects_diff_agg.items():
        logger.info(f"{k}: {v}")

    with open(os.path.join(save_dir,
                           f'eval_object_stats_{fill_type}_cooccurence_understanding.json'), 'w') as f:
        tmp = {
            'mod_single_object_acc': object_means,
            'robustness_patch_mod_normalized_logits': object_robustness,
            'difference_patch_mod_normalized_logits': object_difference,
            'object_counts': Counter(results['single_object']),
            'other_objects_drop': other_objects_diff,
            'other_objects_drop_agg': other_objects_diff_agg,
            'other_objects_patch': other_objects_patch,
            'other_objects_mod': other_objects_mod,
        }
        json.dump(tmp, f)

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
    parser.add_argument('--checkpoint_pth', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--home_dir', type=str, default='/home/mschiappa/SRI/UnderstandingVisualTextModels')
    # Dataset Specific
    parser.add_argument('--annot_dir', type=str,
                        default='/home/mschiappa/SRI/UnderstandingVisualTextModels/new_datasets/context')
    parser.add_argument('--root_image_orig_dir', default='/media/mschiappa/Elements/coco/val2014_background_removed_dataset', type=str)
    parser.add_argument('--root_image_patch_dir', default='/media/mschiappa/Elements/coco/val2014_random_patch_dataset',
                        type=str)
    parser.add_argument('--root_image_mod_dir',
                        default='/media/mschiappa/Elements/coco/val2014_background_removed_and_cooccurrence_dataset', type=str)
    parser.add_argument('--fill_type', type=str, default='noise')
    # parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)


    args = parser.parse_args()

    # Build our save directory for results, must pass the top-level directory results are stored in
    save_dir = os.path.join(args.save_dir, 'cooccurrence_text_eval', args.model.replace('/', '').replace('@', ''))
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

    assert args.fill_type in ['noise', 'black', 'gray', 'scene']
    root_image_orig_dir = os.path.join(args.root_image_orig_dir, 'segmentation_mask', args.fill_type)
    root_image_mod_dir = os.path.join(args.root_image_mod_dir, 'segmentation_mask', args.fill_type)
    root_image_patch_dir = os.path.join(args.root_image_patch_dir, args.fill_type)

    eval(args.model, args.annot_dir, root_image_orig_dir, root_image_mod_dir, root_image_patch_dir, args.fill_type,
         args.num_workers, 1, save_dir)