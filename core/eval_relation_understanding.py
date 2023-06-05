import numpy as np
import datetime
from pathlib import Path
import logging
import argparse
import json
from tqdm import tqdm
from PIL import Image
import torch
import pdb
import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
import torchvision.transforms as transforms
from torchvision.transforms import Compose
import random
from transformers import AutoProcessor
import sys, os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
from UnderstandingVisualTextModels.models.CLIP import clip


class VisualGenomeDataset(Dataset):
    def __init__(self, annot_dir, image_root, switch_triplet='subject'):
        assert switch_triplet in ['subject', 'object'], "Please pass either subject, or object for switch."
        self.switch_triplet = switch_triplet

        self.annotations = json.load(open(os.path.join(annot_dir, f'data/visual_genome/relation_eval_{switch_triplet}.json'), 'r'))

        self.image_ids = list(self.annotations.keys())
        self.image_root = image_root
        self.transform_toTensor = torchvision.transforms.ToTensor()
        self.transform_toPIL = torchvision.transforms.ToPILImage()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        annotation = self.annotations[image_id]

        gt_image = self.load_image(image_id)
        positive_images = [self.load_image(im_id) for im_id in annotation['positive_image_ids']]
        negative_images = [self.load_image(im_id) for im_id in annotation['negative_image_ids']]

        positive_prompt = annotation['positive_prompt']
        negative_prompt = annotation['negative_prompt']
        negative_object = annotation['negative_object']
        positive_object = annotation['positive_object']
        negative_predicate_prompt = annotation['negative_predicate_prompt']

        return gt_image, positive_images, negative_images, positive_prompt, negative_prompt, image_id, negative_object, positive_object, negative_predicate_prompt

    def load_image(self, image_id):
        # Load image that represents the actual relationship
        f = open(os.path.join(self.image_root, str(image_id) + '.jpg'), 'rb')
        image = Image.open(f).convert("RGB")
        image.load()
        f.close()
        return image

    @staticmethod
    def collate_fn(batch):
        gt_image = [x[0] for x in batch]
        positive_images = [x[1] for x in batch]
        negative_images = [x[2] for x in batch]
        positive_prompt = [x[3] for x in batch]
        negative_prompt = [x[4] for x in batch]
        image_id = [x[5] for x in batch]
        negative_obj_prompts = ['A photo of '+x[6]+'.' for x in batch]
        positive_obj_prompts = ['A photo of '+x[7]+'.' for x in batch]
        negative_predicate_prompts = [x[8] for x in batch]
        return gt_image, positive_images, negative_images, positive_prompt, negative_prompt, image_id, negative_obj_prompts, positive_obj_prompts, negative_predicate_prompts


def eval_flava(pbar, results, acc):
    from transformers import FlavaProcessor, FlavaForPreTraining
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FlavaForPreTraining.from_pretrained("facebook/flava-full").to(device)
    preprocess = FlavaProcessor.from_pretrained("facebook/flava-full")

    for gt_image, positive_images, negative_images, positive_prompt, negative_prompt, image_id, negative_obj_prompts, positive_obj_prompts, negative_predicate_prompts in pbar:
        batch_size = len(gt_image)

        prompts = positive_prompt + negative_prompt + positive_obj_prompts + negative_obj_prompts + negative_predicate_prompts

        with torch.no_grad():
            inputs = preprocess(text=prompts, images=gt_image * len(prompts), return_tensors="pt", padding="max_length",
                                max_length=77, return_codebook_pixels=True, return_image_mask=True).to(device)
            outputs = model(**inputs, input_ids_masked=inputs.input_ids)

            image_contrastive_logits = outputs.contrastive_logits_per_image[0]
            rel1_img_vs_rel1_prompt_logits = image_contrastive_logits[0]
            rel1_img_vs_rel3_prompt_logits =  image_contrastive_logits[1]
            result = torch.stack([rel1_img_vs_rel1_prompt_logits, rel1_img_vs_rel3_prompt_logits]).softmax(dim=0)
            results['rel1_image']['rel1_vs_rel3'].append(result[0].cpu().item())
            acc['rel1_image']['rel1_vs_rel3'].append(int(result[0] > result[1]))

            rel1_img_vs_rel2_prompt_logits = image_contrastive_logits[-1]
            result = torch.stack([rel1_img_vs_rel1_prompt_logits, rel1_img_vs_rel2_prompt_logits]).softmax(dim=0)
            results['rel1_image']['rel1_vs_rel2'].append(result[0].cpu().item())
            acc['rel1_image']['rel1_vs_rel2'].append(int(result[0] > result[1]))

            rel1_img_vs_obj1_prompt_logits = image_contrastive_logits[2]
            result = torch.stack([rel1_img_vs_rel1_prompt_logits, rel1_img_vs_obj1_prompt_logits]).softmax(dim=0)
            results['rel1_image']['rel1_vs_obj1'].append(result[0].cpu().item())
            acc['rel1_image']['rel1_vs_obj1'].append(int(result[0] > result[1]))

            # Now look at obj1_img results
            logits = list()
            for obj_image in positive_images[0]:
                inputs = preprocess(text=prompts, images=[obj_image] * len(prompts), return_tensors="pt",
                                    padding="max_length", max_length=77, return_codebook_pixels=True,
                                    return_image_mask=True).to(device)
                inputs.bool_masked_pos.zero_()
                outputs = model(**inputs, input_ids_masked=inputs.input_ids)
                image_contrastive_logits = outputs.contrastive_logits_per_image[0]
                logits.append(image_contrastive_logits)

            image_contrastive_logits = torch.mean(torch.stack(logits), dim=0)

            obj1_img_vs_rel1_prompt_logits = image_contrastive_logits[0]
            obj1_img_vs_obj1_prompt_logits = image_contrastive_logits[2]
            result = torch.stack([obj1_img_vs_obj1_prompt_logits, obj1_img_vs_rel1_prompt_logits]).softmax(dim=0)
            results['obj1_images']['obj1_vs_rel1'].append(result[0].cpu().item())
            acc['obj1_images']['obj1_vs_rel1'].append(int(result[0] > result[1]))

            obj1_img_vs_rel2_prompt_logits = image_contrastive_logits[-1]
            result = torch.stack([obj1_img_vs_obj1_prompt_logits, obj1_img_vs_rel2_prompt_logits]).softmax(dim=0)
            results['obj1_images']['obj1_vs_rel2'].append(result[0].cpu().item())
            acc['obj1_images']['obj1_vs_rel2'].append(int(result[0] > result[1]))

            result = torch.stack([obj1_img_vs_rel1_prompt_logits, obj1_img_vs_rel2_prompt_logits]).softmax(dim=0)
            results['obj1_images']['rel1_vs_rel2'].append(result[0].cpu().item())
            acc['obj1_images']['rel1_vs_rel2'].append(int(result[0] > result[1]))

            obj1_img_vs_rel3_prompt_logits = image_contrastive_logits[1]
            result = torch.stack([obj1_img_vs_rel1_prompt_logits, obj1_img_vs_rel3_prompt_logits]).softmax(dim=0)
            results['obj1_images']['rel1_vs_rel3'].append(result[0].cpu().item())
            acc['obj1_images']['rel1_vs_rel3'].append(int(result[0] > result[1]))

            # Now compare objects only
            obj1_img_vs_obj3_prompt_logits = image_contrastive_logits[3]
            logits = list()
            for obj_image in negative_images[0]:
                inputs = preprocess(text=prompts, images=[obj_image] * len(prompts), return_tensors="pt",
                                    padding="max_length", max_length=77, return_codebook_pixels=True,
                                    return_image_mask=True).to(device)
                inputs.bool_masked_pos.zero_()
                outputs = model(**inputs, input_ids_masked=inputs.input_ids)
                image_contrastive_logits = outputs.contrastive_logits_per_image[0]
                logits.append(image_contrastive_logits)
            image_contrastive_logits = torch.mean(torch.stack(logits), dim=0)
            obj3_img_vs_obj3_prompt_logits = image_contrastive_logits[3]
            obj3_img_vs_obj1_prompt_logits = image_contrastive_logits[2]

            obj_detection_pos = torch.stack(
                [obj1_img_vs_obj1_prompt_logits, obj1_img_vs_obj3_prompt_logits]).softmax(dim=0)
            obj_detection_neg = torch.stack(
                [obj3_img_vs_obj3_prompt_logits, obj3_img_vs_obj1_prompt_logits]).softmax(dim=0)

            results['obj3_images']['obj3_vs_obj1'].append(obj_detection_neg[0].cpu().item())
            acc['obj3_images']['obj3_vs_obj1'].append(int(result[0] > result[1]))

            results['obj1_images']['obj1_vs_obj3'].append(obj_detection_pos[0].cpu().item())
            acc['obj1_images']['obj1_vs_obj3'].append(int(result[0] > result[1]))
    return results, acc


def eval_finetuned_clip(model_weights, alpha, stream, model_name, pbar, results, acc):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    model.eval()

    alpha = 0.2
    theta_0 = model.state_dict()
    print(f'Loading weights of finetuned model and interpolating with original weights with alpha {alpha}')
    checkpoint = torch.load(model_weights)
    theta_1 = checkpoint['final_head']

    # theta_0 = original_weights.state_dict()
    # theta_1 = new_weights.state_dict()
    theta = {
        key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }

    # Update the model acccording to the new weights
    model_name =f'finetuned{stream}_ep20_alpha{alpha}_ViT-B32'
    model.load_state_dict(theta)

    logit_scale = model.logit_scale.exp().detach()
    for gt_image, positive_images, negative_images, positive_prompt, negative_prompt, image_id, negative_obj_prompts, positive_obj_prompts, negative_predicate_prompts in pbar:
        batch_size = len(gt_image)

        prompts = positive_prompt + negative_prompt + positive_obj_prompts + negative_obj_prompts + negative_predicate_prompts
        gt_images = torch.stack([preprocess(image) for image in gt_image]).to(device)

        with torch.no_grad():
            # Get features
            visual_features = model.encode_image(gt_images)
            if isinstance(visual_features, tuple):
                visual_features = visual_features[0]

            text_features = model.encode_text(prompts)
            if isinstance(text_features, tuple):
                text_features = text_features[0]

            text_features = normalize(text_features)
            visual_features = normalize(visual_features)

        # Split features
        rel1_img_features = visual_features[:batch_size]
        dims = visual_features.shape[-1]


        rel1_prompt_txt_features = text_features[:batch_size]
        rel3_prompt_txt_features = text_features[batch_size: batch_size * 2]
        # obj1_txt_features = text_features[batch_size * 2: batch_size * 3]
        # obj3_txt_features = text_features[batch_size * 3: batch_size * 4]
        rel2_prompt_txt_features = text_features[batch_size * 4: batch_size * 5]

        # Get similarity compared to all prompts, gt -> rel1, gt+diffobj->rel2 gt_gt+
        """
        rel1 is <subject1> <pred1> <object1>
        rel2 is <subject1> <pred2> <object1>
        rel3 is <subject2> <pred1> <object1>
        obj1 is <subject1>
        obj3 is <subject2>
        """
        for batch_idx, im_f in enumerate(rel1_img_features):
            # Compare gt image to prompts with gt objects and an object switched
            rel1_img_vs_rel1_prompt_logits = get_logits(logit_scale, im_f, rel1_prompt_txt_features[batch_idx])
            rel1_img_vs_rel3_prompt_logits = get_logits(logit_scale, im_f, rel3_prompt_txt_features[batch_idx])
            result = torch.stack([rel1_img_vs_rel1_prompt_logits, rel1_img_vs_rel3_prompt_logits]).softmax(dim=0)
            results['rel1_image']['rel1_vs_rel3'].append(result[0].cpu().item())
            acc['rel1_image']['rel1_vs_rel3'].append(int(result[0] > result[1]))

            # Compare gt image to prompt with gt predicate and predicate switched
            rel1_img_vs_rel2_prompt_logits = get_logits(logit_scale, im_f, rel2_prompt_txt_features[batch_idx])
            result = torch.stack([rel1_img_vs_rel1_prompt_logits, rel1_img_vs_rel2_prompt_logits]).softmax(dim=0)
            results['rel1_image']['rel1_vs_rel2'].append(result[0].cpu().item())
            acc['rel1_image']['rel1_vs_rel2'].append(int(result[0] > result[1]))

            # Compare performance when full prompt to full objects
            # rel1_img_vs_obj1_prompt_logits = get_logits(logit_scale, im_f, obj1_txt_features[batch_idx])
            # result = torch.stack([rel1_img_vs_rel1_prompt_logits, rel1_img_vs_obj1_prompt_logits]).softmax(dim=0)
            # results['rel1_image']['rel1_vs_obj1'].append(result[0].cpu().item())
            # acc['rel1_image']['rel1_vs_obj1'].append(int(result[0] > result[1]))

            # # Get positive images comparison, get mean similarity
            # obj1_img_vs_rel1_prompt_logits = torch.mean(get_logits(logit_scale, obj1_img_features[batch_idx],
            #                                                        rel1_prompt_txt_features[batch_idx].unsqueeze(0)))
            # obj1_img_vs_obj1_prompt_logits = torch.mean(get_logits(logit_scale, obj1_img_features[batch_idx],
            #                                                        obj1_txt_features[batch_idx].unsqueeze(0)))
            #
            # # Comparing the positive images to the gt prompt, even though other object not present
            # result = torch.stack([obj1_img_vs_obj1_prompt_logits, obj1_img_vs_rel1_prompt_logits]).softmax(dim=0)
            # results['obj1_images']['obj1_vs_rel1'].append(result[0].cpu().item())
            # acc['obj1_images']['obj1_vs_rel1'].append(int(result[0] > result[1]))
            #
            # # Comparing the positive images to the gt prompt, even though other object not present
            # # result = torch.stack([obj1_img_vs_obj1_prompt_logits, obj1_img_vs_rel1_prompt_logits]).softmax(dim=0)
            # # results['obj1_images']['obj1_vs_rel1'].append(result[0].cpu().item())
            # # acc['obj1_images']['obj1_vs_rel1'].append(int(result[0] > result[1]))
            #
            # # This is comparing the same but with an incorrect predicate
            # obj1_img_vs_rel2_prompt_logits = torch.mean(get_logits(logit_scale, obj1_img_features[batch_idx],
            #                                                        rel2_prompt_txt_features[batch_idx].unsqueeze(
            #                                                            0)))
            #
            # result = torch.stack([obj1_img_vs_obj1_prompt_logits, obj1_img_vs_rel2_prompt_logits]).softmax(dim=0)
            # results['obj1_images']['obj1_vs_rel2'].append(result[0].cpu().item())
            # acc['obj1_images']['obj1_vs_rel2'].append(int(result[0] > result[1]))
            #
            # result = torch.stack([obj1_img_vs_rel1_prompt_logits, obj1_img_vs_rel2_prompt_logits]).softmax(dim=0)
            # results['obj1_images']['rel1_vs_rel2'].append(result[0].cpu().item())
            # acc['obj1_images']['rel1_vs_rel2'].append(int(result[0] > result[1]))
            #
            # # FIXXXX THISSSS This is comparing object with the ground truth relation with different object
            # obj1_img_vs_rel3_prompt_logits = torch.mean(get_logits(logit_scale, obj1_img_features[batch_idx],
            #                                                        rel3_prompt_txt_features[batch_idx].unsqueeze(
            #                                                            0)))
            # result = torch.stack([obj1_img_vs_rel1_prompt_logits, obj1_img_vs_rel3_prompt_logits]).softmax(dim=0)
            # results['obj1_images']['rel1_vs_rel3'].append(result[0].cpu().item())
            # acc['obj1_images']['rel1_vs_rel3'].append(int(result[0] > result[1]))
            #
            # #####
            # # Get positive images comparison, get mean similarity
            # # neg_positive_prompt_logits = torch.mean(get_logits(logit_scale, obj3_img_features[batch_idx],
            # #                                                    rel3_prompt_txt_features[batch_idx].unsqueeze(
            # #                                                        0)))
            #
            # # Logits of the images with the object switched with correct object prompt
            # obj3_img_vs_obj3_prompt_logits = torch.mean(get_logits(logit_scale, obj3_img_features[batch_idx],
            #                                                        obj3_txt_features[batch_idx].unsqueeze(0)))
            #
            # # Now lets see how good the model is at actual object recognioton
            # obj1_img_vs_obj3_prompt_logits = torch.mean(get_logits(logit_scale, obj1_img_features[batch_idx],
            #                                                        obj3_txt_features[batch_idx].unsqueeze(0)))
            # obj_detection_pos = torch.stack([obj1_img_vs_obj1_prompt_logits, obj1_img_vs_obj3_prompt_logits]).softmax(dim=0)
            #
            # obj3_img_vs_obj1_prompt_logits = torch.mean(get_logits(logit_scale, obj3_img_features[batch_idx],
            #                                                        obj1_txt_features[batch_idx].unsqueeze(0)))
            #
            # obj_detection_neg = torch.stack([obj3_img_vs_obj3_prompt_logits, obj3_img_vs_obj1_prompt_logits]).softmax(dim=0)
            #
            # results['obj3_images']['obj3_vs_obj1'].append(obj_detection_neg[0].cpu().item())
            # acc['obj3_images']['obj3_vs_obj1'].append(int(result[0] > result[1]))
            #
            # results['obj1_images']['obj1_vs_obj3'].append(obj_detection_pos[0].cpu().item())
            # acc['obj1_images']['obj1_vs_obj3'].append(int(result[0] > result[1]))
    return results, acc, model_name

def eval_clip(model_name, pbar, results, acc):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model_type = model_name.split('_')[-1]
    logger.info(f"Building {clip_model_type} based CLIP model...")

    model_name = clip_model_type.replace("/", "")
    model, preprocess = clip.load(clip_model_type, device)
    model.eval()
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)

    logit_scale = model.logit_scale.exp().detach()
    for gt_image, positive_images, negative_images, positive_prompt, negative_prompt, image_id, negative_obj_prompts, positive_obj_prompts, negative_predicate_prompts in pbar:
        batch_size = len(gt_image)
        # TEMPORARY FOR TESTING
        # gt_image[-1].save('test/gt_image.jpg')
        # for idx, img in enumerate(positive_images[-1]): img.save(f'test/positive_{idx}.jpg')
        # for idx, img in enumerate(negative_images[-1]): img.save(f'test/negative_{idx}.jpg')
        #0
        prompts = positive_prompt + negative_prompt + positive_obj_prompts + negative_obj_prompts + negative_predicate_prompts
        gt_images = torch.stack([preprocess(image) for image in gt_image]).to(device)
        positive_images = torch.stack([preprocess(image) for images in positive_images for image in images]).to(
            device)
        negative_images = torch.stack([preprocess(image) for images in negative_images for image in images]).to(
            device)
        full_images = torch.cat([gt_images, positive_images, negative_images])
        with torch.no_grad():
            # Get features
            visual_features = model.encode_image(full_images)
            if isinstance(visual_features, tuple):
                visual_features = visual_features[0]

            text_features = model.encode_text(prompts)
            if isinstance(text_features, tuple):
                text_features = text_features[0]

            text_features = normalize(text_features)
            visual_features = normalize(visual_features)

        # Split features
        rel1_img_features = visual_features[:batch_size]
        dims = visual_features.shape[-1]
        obj1_img_features = visual_features[batch_size: batch_size + (batch_size * 10)].reshape(batch_size, 10, dims)
        obj3_img_features = visual_features[
                            (batch_size + (batch_size * 10)): (batch_size + (batch_size * 10)) + (batch_size * 10)].reshape(
            batch_size, 10, dims)

        # tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

        rel1_prompt_txt_features = text_features[:batch_size]
        rel3_prompt_txt_features = text_features[batch_size: batch_size * 2]
        obj1_txt_features = text_features[batch_size * 2: batch_size * 3]
        obj3_txt_features = text_features[batch_size * 3: batch_size * 4]
        rel2_prompt_txt_features = text_features[batch_size * 4: batch_size * 5]

        # Get similarity compared to all prompts, gt -> rel1, gt+diffobj->rel2 gt_gt+
        """
        rel1 is <subject1> <pred1> <object1>
        rel2 is <subject1> <pred2> <object1>
        rel3 is <subject2> <pred1> <object1>
        obj1 is <subject1>
        obj3 is <subject2>
        """
        for batch_idx, im_f in enumerate(rel1_img_features):
            # Compare gt image to prompts with gt objects and an object switched
            rel1_img_vs_rel1_prompt_logits = get_logits(logit_scale, im_f, rel1_prompt_txt_features[batch_idx])
            rel1_img_vs_rel3_prompt_logits = get_logits(logit_scale, im_f, rel3_prompt_txt_features[batch_idx])
            result = torch.stack([rel1_img_vs_rel1_prompt_logits, rel1_img_vs_rel3_prompt_logits]).softmax(dim=0)
            results['rel1_image']['rel1_vs_rel3'].append(result[0].cpu().item())
            acc['rel1_image']['rel1_vs_rel3'].append(int(result[0] > result[1]))

            # Compare gt image to prompt with gt predicate and predicate switched
            rel1_img_vs_rel2_prompt_logits = get_logits(logit_scale, im_f, rel2_prompt_txt_features[batch_idx])
            result = torch.stack([rel1_img_vs_rel1_prompt_logits, rel1_img_vs_rel2_prompt_logits]).softmax(dim=0)
            results['rel1_image']['rel1_vs_rel2'].append(result[0].cpu().item())
            acc['rel1_image']['rel1_vs_rel2'].append(int(result[0] > result[1]))

            # Compare performance when full prompt to full objects
            rel1_img_vs_obj1_prompt_logits = get_logits(logit_scale, im_f, obj1_txt_features[batch_idx])
            result = torch.stack([rel1_img_vs_rel1_prompt_logits, rel1_img_vs_obj1_prompt_logits]).softmax(dim=0)
            results['rel1_image']['rel1_vs_obj1'].append(result[0].cpu().item())
            acc['rel1_image']['rel1_vs_obj1'].append(int(result[0] > result[1]))

            # Get positive images comparison, get mean similarity
            obj1_img_vs_rel1_prompt_logits = torch.mean(get_logits(logit_scale, obj1_img_features[batch_idx],
                                                                   rel1_prompt_txt_features[batch_idx].unsqueeze(0)))
            obj1_img_vs_obj1_prompt_logits = torch.mean(get_logits(logit_scale, obj1_img_features[batch_idx],
                                                                   obj1_txt_features[batch_idx].unsqueeze(0)))

            # Comparing the positive images to the gt prompt, even though other object not present
            result = torch.stack([obj1_img_vs_obj1_prompt_logits, obj1_img_vs_rel1_prompt_logits]).softmax(dim=0)
            results['obj1_images']['obj1_vs_rel1'].append(result[0].cpu().item())
            acc['obj1_images']['obj1_vs_rel1'].append(int(result[0] > result[1]))

            # Comparing the positive images to the gt prompt, even though other object not present
            # result = torch.stack([obj1_img_vs_obj1_prompt_logits, obj1_img_vs_rel1_prompt_logits]).softmax(dim=0)
            # results['obj1_images']['obj1_vs_rel1'].append(result[0].cpu().item())
            # acc['obj1_images']['obj1_vs_rel1'].append(int(result[0] > result[1]))

            # This is comparing the same but with an incorrect predicate
            obj1_img_vs_rel2_prompt_logits = torch.mean(get_logits(logit_scale, obj1_img_features[batch_idx],
                                                                   rel2_prompt_txt_features[batch_idx].unsqueeze(
                                                                       0)))

            result = torch.stack([obj1_img_vs_obj1_prompt_logits, obj1_img_vs_rel2_prompt_logits]).softmax(dim=0)
            results['obj1_images']['obj1_vs_rel2'].append(result[0].cpu().item())
            acc['obj1_images']['obj1_vs_rel2'].append(int(result[0] > result[1]))

            result = torch.stack([obj1_img_vs_rel1_prompt_logits, obj1_img_vs_rel2_prompt_logits]).softmax(dim=0)
            results['obj1_images']['rel1_vs_rel2'].append(result[0].cpu().item())
            acc['obj1_images']['rel1_vs_rel2'].append(int(result[0] > result[1]))

            # FIXXXX THISSSS This is comparing object with the ground truth relation with different object
            obj1_img_vs_rel3_prompt_logits = torch.mean(get_logits(logit_scale, obj1_img_features[batch_idx],
                                                                   rel3_prompt_txt_features[batch_idx].unsqueeze(
                                                                       0)))
            result = torch.stack([obj1_img_vs_rel1_prompt_logits, obj1_img_vs_rel3_prompt_logits]).softmax(dim=0)
            results['obj1_images']['rel1_vs_rel3'].append(result[0].cpu().item())
            acc['obj1_images']['rel1_vs_rel3'].append(int(result[0] > result[1]))

            #####
            # Get positive images comparison, get mean similarity
            # neg_positive_prompt_logits = torch.mean(get_logits(logit_scale, obj3_img_features[batch_idx],
            #                                                    rel3_prompt_txt_features[batch_idx].unsqueeze(
            #                                                        0)))

            # Logits of the images with the object switched with correct object prompt
            obj3_img_vs_obj3_prompt_logits = torch.mean(get_logits(logit_scale, obj3_img_features[batch_idx],
                                                                   obj3_txt_features[batch_idx].unsqueeze(0)))

            # Now lets see how good the model is at actual object recognioton
            obj1_img_vs_obj3_prompt_logits = torch.mean(get_logits(logit_scale, obj1_img_features[batch_idx],
                                                                   obj3_txt_features[batch_idx].unsqueeze(0)))
            obj_detection_pos = torch.stack([obj1_img_vs_obj1_prompt_logits, obj1_img_vs_obj3_prompt_logits]).softmax(dim=0)

            obj3_img_vs_obj1_prompt_logits = torch.mean(get_logits(logit_scale, obj3_img_features[batch_idx],
                                                                   obj1_txt_features[batch_idx].unsqueeze(0)))

            obj_detection_neg = torch.stack([obj3_img_vs_obj3_prompt_logits, obj3_img_vs_obj1_prompt_logits]).softmax(dim=0)

            results['obj3_images']['obj3_vs_obj1'].append(obj_detection_neg[0].cpu().item())
            acc['obj3_images']['obj3_vs_obj1'].append(int(result[0] > result[1]))

            results['obj1_images']['obj1_vs_obj3'].append(obj_detection_pos[0].cpu().item())
            acc['obj1_images']['obj1_vs_obj3'].append(int(result[0] > result[1]))
    return results, acc, model_name


def eval_bridgetower(pbar, results, acc):
    """
        This is incredibly slow, very very depressing.
        Trying to increase speed, requires > 48GB of capacity which is ridiculuos
        https://huggingface.co/docs/transformers/main/en/model_doc/bridgetower#overview
        :param class_prompts:
        :param pbar:
        :param results:
        :return:
        """
    from transformers import BridgeTowerForImageAndTextRetrieval, BridgeTowerProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "BridgeTower/bridgetower-large-itm-mlm"
    preprocess = BridgeTowerProcessor.from_pretrained(model_id)
    model = BridgeTowerForImageAndTextRetrieval.from_pretrained(model_id).to(device)
    model.eval()

    for gt_image, positive_images, negative_images, positive_prompt, negative_prompt, image_id, negative_obj_prompts, positive_obj_prompts, negative_predicate_prompts in pbar:
        batch_size = len(gt_image)

        prompts = positive_prompt + negative_prompt + positive_obj_prompts + negative_obj_prompts + negative_predicate_prompts

        with torch.no_grad():
            text_inputs = preprocess(images=gt_image * len(prompts),
                                     text=prompts,
                                     return_tensors="pt",
                                     padding="max_length",
                                     truncation=False,
                                     max_length=40).to(device)

            image_contrastive_logits = model(**text_inputs.to(device)).logits[:, 1] #.t()

        rel1_img_vs_rel1_prompt_logits = image_contrastive_logits[0]
        rel1_img_vs_rel3_prompt_logits = image_contrastive_logits[1]
        result = torch.stack([rel1_img_vs_rel1_prompt_logits, rel1_img_vs_rel3_prompt_logits]).softmax(dim=0)
        results['rel1_image']['rel1_vs_rel3'].append(result[0].cpu().item())
        acc['rel1_image']['rel1_vs_rel3'].append(int(result[0] > result[1]))

        rel1_img_vs_rel2_prompt_logits = image_contrastive_logits[-1]
        result = torch.stack([rel1_img_vs_rel1_prompt_logits, rel1_img_vs_rel2_prompt_logits]).softmax(dim=0)
        results['rel1_image']['rel1_vs_rel2'].append(result[0].cpu().item())
        acc['rel1_image']['rel1_vs_rel2'].append(int(result[0] > result[1]))

        rel1_img_vs_obj1_prompt_logits = image_contrastive_logits[2]
        result = torch.stack([rel1_img_vs_rel1_prompt_logits, rel1_img_vs_obj1_prompt_logits]).softmax(dim=0)
        results['rel1_image']['rel1_vs_obj1'].append(result[0].cpu().item())
        acc['rel1_image']['rel1_vs_obj1'].append(int(result[0] > result[1]))

        # with torch.no_grad():
        #     # Now look at obj1_img results
        #     logits = list()
        #     for obj_image in positive_images[0]:
        #         text_inputs = preprocess(images=[obj_image] * len(prompts),
        #                                  text=prompts,
        #                                  return_tensors="pt",
        #                                  padding="max_length",
        #                                  truncation=False,
        #                                  max_length=40).to(device)
        #         image_contrastive_logits = model(**text_inputs.to(device)).logits[:, 1].t()
        #         logits.append(image_contrastive_logits)
        with torch.no_grad():
            inputs =  preprocess(images=positive_images[0],
                                          text=prompts[0], # + prompts[2],
                                          return_tensors="pt",
                                          padding="max_length",
                                          truncation=False,
                                          max_length=40).to(device)
            obj1_img_vs_rel1_prompt_logits = model(**inputs.to(device)).logits[:, 1].t()
            inputs = preprocess(images=positive_images[0],
                                text=prompts[2],
                                return_tensors="pt",
                                padding="max_length",
                                truncation=False,
                                max_length=40).to(device)
            obj1_img_vs_obj1_prompt_logits = model(**inputs.to(device)).logits[:, 1].t()
        # obj1_img_vs_rel1_prompt_logits = image_contrastive_logits[0]
        # obj1_img_vs_obj1_prompt_logits = image_contrastive_logits[2]
        result = torch.stack([obj1_img_vs_obj1_prompt_logits, obj1_img_vs_rel1_prompt_logits]).softmax(dim=0)
        result = torch.mean(result, dim=1)
        results['obj1_images']['obj1_vs_rel1'].append(result[0].cpu().item())
        acc['obj1_images']['obj1_vs_rel1'].append(int(result[0] > result[1]))

        # obj1_img_vs_rel2_prompt_logits = image_contrastive_logits[-1]
        # result = torch.stack([obj1_img_vs_obj1_prompt_logits, obj1_img_vs_rel2_prompt_logits]).softmax(dim=0)
        # results['obj1_images']['obj1_vs_rel2'].append(result[0].cpu().item())
        # acc['obj1_images']['obj1_vs_rel2'].append(int(result[0] > result[1]))
        #
        # result = torch.stack([obj1_img_vs_rel1_prompt_logits, obj1_img_vs_rel2_prompt_logits]).softmax(dim=0)
        # results['obj1_images']['rel1_vs_rel2'].append(result[0].cpu().item())
        # acc['obj1_images']['rel1_vs_rel2'].append(int(result[0] > result[1]))
        #
        # obj1_img_vs_rel3_prompt_logits = image_contrastive_logits[1]
        # result = torch.stack([obj1_img_vs_rel1_prompt_logits, obj1_img_vs_rel3_prompt_logits]).softmax(dim=0)
        # results['obj1_images']['rel1_vs_rel3'].append(result[0].cpu().item())
        # acc['obj1_images']['rel1_vs_rel3'].append(int(result[0] > result[1]))

        # with torch.no_grad():
        #     # Now compare objects only
        #     obj1_img_vs_obj3_prompt_logits = image_contrastive_logits[3]
        #     logits = list()
        #     for obj_image in negative_images[0]:
        #         inputs = preprocess(images=[obj_image] * len(prompts),
        #                             text=prompts,
        #                             return_tensors="pt",
        #                             padding="max_length",
        #                             truncation=False,
        #                             max_length=40).to(device)
        #         image_contrastive_logits = model(**inputs.to(device)).logits[:, 1].cpu().t()
        #         logits.append(image_contrastive_logits)

        # image_contrastive_logits = torch.mean(torch.stack(logits), dim=0)
        # obj3_img_vs_obj3_prompt_logits = image_contrastive_logits[3]
        # obj3_img_vs_obj1_prompt_logits = image_contrastive_logits[2]

        # obj_detection_pos = torch.stack(
        #     [obj1_img_vs_obj1_prompt_logits, obj1_img_vs_obj3_prompt_logits]).softmax(dim=0)
        # obj_detection_neg = torch.stack(
        #     [obj3_img_vs_obj3_prompt_logits, obj3_img_vs_obj1_prompt_logits]).softmax(dim=0)

        # results['obj3_images']['obj3_vs_obj1'].append(obj_detection_neg[0].item())
        # acc['obj3_images']['obj3_vs_obj1'].append(int(result[0] > result[1]))

        # results['obj1_images']['obj1_vs_obj3'].append(obj_detection_pos[0].item())
        # acc['obj1_images']['obj1_vs_obj3'].append(int(result[0] > result[1]))
    return results, acc


def eval_vilt(pbar, results, acc):
    # https://arxiv.org/pdf/2102.03334.pdf
    from transformers import ViltProcessor, ViltForImageAndTextRetrieval
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
    model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")
    model.to(device)

    for gt_image, positive_images, negative_images, positive_prompt, negative_prompt, image_id, negative_obj_prompts, positive_obj_prompts, negative_predicate_prompts in pbar:
        batch_size = len(gt_image)

        prompts = positive_prompt + negative_prompt + positive_obj_prompts + negative_obj_prompts + negative_predicate_prompts

        with torch.no_grad():
            inputs = preprocess(images=gt_image * len(prompts),
                                text=prompts,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=False,
                                max_length=40).to(device)
            outputs = model(**inputs)

            image_contrastive_logits = outputs.logits.t()[0]
            rel1_img_vs_rel1_prompt_logits = image_contrastive_logits[0]
            rel1_img_vs_rel3_prompt_logits =  image_contrastive_logits[1]
            result = torch.stack([rel1_img_vs_rel1_prompt_logits, rel1_img_vs_rel3_prompt_logits]).softmax(dim=0)
            results['rel1_image']['rel1_vs_rel3'].append(result[0].cpu().item())
            acc['rel1_image']['rel1_vs_rel3'].append(int(result[0] > result[1]))

            rel1_img_vs_rel2_prompt_logits = image_contrastive_logits[-1]
            result = torch.stack([rel1_img_vs_rel1_prompt_logits, rel1_img_vs_rel2_prompt_logits]).softmax(dim=0)
            results['rel1_image']['rel1_vs_rel2'].append(result[0].cpu().item())
            acc['rel1_image']['rel1_vs_rel2'].append(int(result[0] > result[1]))

            rel1_img_vs_obj1_prompt_logits = image_contrastive_logits[2]
            result = torch.stack([rel1_img_vs_rel1_prompt_logits, rel1_img_vs_obj1_prompt_logits]).softmax(dim=0)
            results['rel1_image']['rel1_vs_obj1'].append(result[0].cpu().item())
            acc['rel1_image']['rel1_vs_obj1'].append(int(result[0] > result[1]))

            # # Now look at obj1_img results
            # logits = list()
            # for obj_image in positive_images[0]:
            #     inputs = preprocess(images=[obj_image] * len(prompts),
            #                         text=prompts,
            #                         return_tensors="pt",
            #                         padding="max_length",
            #                         truncation=False,
            #                         max_length=40).to(device)
            #
            #     outputs = model(**inputs)
            #     image_contrastive_logits = outputs.logits.t()[0]
            #     logits.append(image_contrastive_logits)
            #
            # image_contrastive_logits = torch.mean(torch.stack(logits), dim=0)
            #
            # obj1_img_vs_rel1_prompt_logits = image_contrastive_logits[0]
            # obj1_img_vs_obj1_prompt_logits = image_contrastive_logits[2]
            # result = torch.stack([obj1_img_vs_obj1_prompt_logits, obj1_img_vs_rel1_prompt_logits]).softmax(dim=0)
            # results['obj1_images']['obj1_vs_rel1'].append(result[0].cpu().item())
            # acc['obj1_images']['obj1_vs_rel1'].append(int(result[0] > result[1]))
            #
            # obj1_img_vs_rel2_prompt_logits = image_contrastive_logits[-1]
            # result = torch.stack([obj1_img_vs_obj1_prompt_logits, obj1_img_vs_rel2_prompt_logits]).softmax(dim=0)
            # results['obj1_images']['obj1_vs_rel2'].append(result[0].cpu().item())
            # acc['obj1_images']['obj1_vs_rel2'].append(int(result[0] > result[1]))
            #
            # result = torch.stack([obj1_img_vs_rel1_prompt_logits, obj1_img_vs_rel2_prompt_logits]).softmax(dim=0)
            # results['obj1_images']['rel1_vs_rel2'].append(result[0].cpu().item())
            # acc['obj1_images']['rel1_vs_rel2'].append(int(result[0] > result[1]))
            #
            # obj1_img_vs_rel3_prompt_logits = image_contrastive_logits[1]
            # result = torch.stack([obj1_img_vs_rel1_prompt_logits, obj1_img_vs_rel3_prompt_logits]).softmax(dim=0)
            # results['obj1_images']['rel1_vs_rel3'].append(result[0].cpu().item())
            # acc['obj1_images']['rel1_vs_rel3'].append(int(result[0] > result[1]))
            #
            # # Now compare objects only
            # obj1_img_vs_obj3_prompt_logits = image_contrastive_logits[3]
            # logits = list()
            # for obj_image in negative_images[0]:
            #     inputs = preprocess(images=[obj_image] * len(prompts),
            #                         text=prompts,
            #                         return_tensors="pt",
            #                         padding="max_length",
            #                         truncation=False,
            #                         max_length=40).to(device)
            #
            #     outputs = model(**inputs)
            #     image_contrastive_logits = outputs.logits.t()[0]
            #     logits.append(image_contrastive_logits)
            #
            # image_contrastive_logits = torch.mean(torch.stack(logits), dim=0)
            # obj3_img_vs_obj3_prompt_logits = image_contrastive_logits[3]
            # obj3_img_vs_obj1_prompt_logits = image_contrastive_logits[2]
            #
            # obj_detection_pos = torch.stack(
            #     [obj1_img_vs_obj1_prompt_logits, obj1_img_vs_obj3_prompt_logits]).softmax(dim=0)
            # obj_detection_neg = torch.stack(
            #     [obj3_img_vs_obj3_prompt_logits, obj3_img_vs_obj1_prompt_logits]).softmax(dim=0)
            #
            # results['obj3_images']['obj3_vs_obj1'].append(obj_detection_neg[0].cpu().item())
            # acc['obj3_images']['obj3_vs_obj1'].append(int(result[0] > result[1]))
            #
            # results['obj1_images']['obj1_vs_obj3'].append(obj_detection_pos[0].cpu().item())
            # acc['obj1_images']['obj1_vs_obj3'].append(int(result[0] > result[1]))
            # pbar.set_postfix({'Rel1_vs_Rel2_conf'})
    return results, acc


def eval(model_name, annot_dir, root_image_dir, num_workers, batch_size, save_dir, model_weights,
         alpha, stream):
    """
    This will evaluate the confusion on non-sensible combinations versus sensible ones.
    :param model:
    :param model_name:
    :param annot_dir:
    :param root_image_dir:
    :param num_workers:
    :param batch_size:
    :return:
    """
    logger.info("Building dataset...")
    dataset = VisualGenomeDataset(annot_dir, root_image_dir)
    dataloader = DataLoader(dataset, num_workers=num_workers, drop_last=False, batch_size=batch_size,
                            collate_fn=dataset.collate_fn)
    logger.info(f"Done building dataset with {len(dataset)} total samples and {len(dataloader)} batches.")
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Running for {model_name}")

    # Create our dictionary of results.
    columns = ['rel1_vs_rel3',
               'rel1_vs_obj1',
               'rel1_vs_rel2',
               'obj1_vs_rel1',
               'obj2_vs_rel2',
               'obj3_vs_obj1',
               'obj1_vs_obj3',
               'obj1_vs_rel2']

    results = dict()
    results['rel1_image'] = dict()
    results['obj1_images'] = dict()
    results['obj3_images'] = dict()

    for key in results.keys():
        for col in columns:
            results[key][col] = list()

    acc = dict()
    acc['rel1_image'] = dict()
    acc['obj1_images'] = dict()
    acc['obj3_images'] = dict()

    for key in acc.keys():
        for col in columns:
            acc[key][col] = list()

    if model_name == 'flava':
        results, acc = eval_flava(pbar, results, acc)
    elif 'clip' in model_name and 'finetune' in model_name:
        results, acc, model_name = eval_finetuned_clip(model_weights, alpha, stream, model_name, pbar, results, acc)
    elif 'clip' in model_name:
        results, acc, model_name = eval_clip(model_name, pbar, results, acc)
    elif model_name == 'vilt':
        results, acc = eval_vilt(pbar, results, acc)
    elif model_name == 'bridgetower':
        results, acc = eval_bridgetower(pbar, results, acc)

    save_dir = os.path.join(save_dir, 'results', model_name)
    if not os.path.exists(save_dir):
        Path(save_dir).mkdir(parents=True)

    with open(os.path.join(save_dir, 'eval_relation_understanding.json'), 'w') as f:
        json.dump({'rel_diff': results, 'acc': acc}, f)

    logger.info("-----------------------")
    logger.info("Softmax:")
    for key, vals in results.items():
        for comparison, scores in vals.items():
            if len(scores) > 0:
                results[key][comparison] = round(np.mean(scores), 4)
                logger.info(f"Results for {key} for comparison {comparison}: {results[key][comparison]}")

    logger.info("-----------------------")
    logger.info("Accuracy:")
    for key, vals in acc.items():
        for comparison, scores in vals.items():
            if len(scores) > 0:
                acc[key][comparison] = round((np.sum(scores) / len(scores)).item(), 4)
                logger.info(f"Results for {key} for comparison {comparison}: {acc[key][comparison]}")
    logger.info("-----------------------")

    with open(os.path.join(save_dir, 'eval_aggregation_relation_understanding.json'), 'w') as f:
        json.dump({'rel_diff': results, 'acc': acc}, f)


def get_mean_similarity(logit_scale, visual_features, text_features):
    logits = get_logits(logit_scale, visual_features, text_features)
    logits = torch.mean(logits, dim=1)
    return logits


@torch.jit.script
def normalize(x):
    return x / x.norm(dim=-1, keepdim=True, p=2)


@torch.jit.script
def get_logits(logit_scale, imf_i, text_f):
    # return logit_scale * imf_i @ text_f.t()
    return logit_scale * text_f @ imf_i.t()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs a Probe-R in LLM+ models.')
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--checkpoint_pth', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--home_dir', type=str, default='/home/schiappa/SRI/UnderstandingVisualTextModels')
    parser.add_argument('--model_weights', type=str,
                        default='/home/schiappa/SRI/UnderstandingVisualTextModels/output/ViT-B32/finetune/lr0.001_wd0.0001/run1/clip_pt_ep10.pth')
    # Dataset Specific
    parser.add_argument('--annot_dir', type=str, default='/home/c3-0/datasets/visual_genome/annotations')
    parser.add_argument('--root_image_dir', default='/home/c3-0/datasets/visual_genome/all_images', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--stream', type=str, default='vt')

    args = parser.parse_args()

    # Build our save directory for results, must pass the top-level directory results are stored in
    save_dir = os.path.join(args.save_dir, 'relational_text_eval', args.model.replace('/', '').replace('@', ''))
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval(args.model, args.annot_dir, args.root_image_dir, args.num_workers, 1, save_dir, args.model_weights, args.alpha,
         args.stream)