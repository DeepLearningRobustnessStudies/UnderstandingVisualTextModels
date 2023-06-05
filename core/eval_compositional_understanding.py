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
import random
from torch.utils.data import Dataset, DataLoader

import sys, os, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
from UnderstandingVisualTextModels.models.CLIP import clip


class COCOCompositionalDataset(Dataset):
    def __init__(self, annot_dir, image_root, switch='composition', subset=False):
        assert switch in ['object', 'composition'], "Please pass either `composition` or `object` for variable `switch`"
        self.switch = switch
        self.image_root = image_root
        pth = 'data_compositional_switch.json' if switch == 'composition' else 'data_object_switch.json'
        pth = os.path.join(annot_dir, pth)
        self.annotations = json.load(open(pth, 'r'))

    def __len__(self):
        return len(self.annotations)

    def load_image(self, image_id):
        # Load image that represents the actual relationship
        pth = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
        f = open(os.path.join(self.image_root, pth), 'rb')
        image = Image.open(f).convert("RGB")
        image.load()
        f.close()
        return image

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        img1_id = annotation['img1_id']
        image1 = self.load_image(img1_id)

        img2_id = annotation['img2_id']
        image2 = self.load_image(img2_id)

        prompt1 = annotation['img1_prompt']
        prompt2 = annotation['img2_prompt']

        return image1, image2, prompt1, prompt2

    def collate_fn(self, batch):
        images1 =  [x[0] for x in batch]
        images2 = [x[1] for x in batch]
        prompt1 = [x[2] for x in batch]
        prompt2 = [x[3] for x in batch]
        return images1, images2, prompt1, prompt2


def eval_clip(model_name, pbar, results, acc):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model_type = model_name.split('_')[-1]
    logger.info(f"Building {clip_model_type} based CLIP model...")

    model, preprocess = clip.load(clip_model_type, device)
    model.eval()
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)

    logit_scale = model.logit_scale.exp().detach()
    for image1, image2, prompt1, prompt2 in pbar:
        batch_size = len(image1)
        with torch.no_grad():
            images = torch.stack([preprocess(image) for image in image1 + image2]).to(device)

            # Get features
            visual_features = model.encode_image(images)
            if isinstance(visual_features, tuple):
                visual_features = visual_features[0]

            text_features = model.encode_text(prompt1 + prompt2)
            if isinstance(text_features, tuple):
                text_features = text_features[0]

        visual_features = normalize(visual_features)
        text_features = normalize(text_features)

        image1_features = visual_features[:batch_size]
        image2_features = visual_features[batch_size:]

        prompt1_features = text_features[:batch_size]
        prompt2_features = text_features[batch_size:]

        batch_idx = 0
        # for batch_idx in range(batch_size):
        image1_vs_prompt1 = get_logits(logit_scale, image1_features[batch_idx], prompt1_features[batch_idx])
        image1_vs_prompt2 = get_logits(logit_scale, image1_features[batch_idx], prompt2_features[batch_idx])

        result = torch.stack([image1_vs_prompt1, image1_vs_prompt2]).softmax(dim=0)
        results['image1']['prompt1'].append(result[0].item())
        results['image1']['prompt2'].append(result[1].item())

        image2_vs_prompt2 = get_logits(logit_scale, image2_features[batch_idx], prompt2_features[batch_idx])
        image2_vs_prompt1 = get_logits(logit_scale, image2_features[batch_idx], prompt1_features[batch_idx])
        result = torch.stack([image2_vs_prompt2, image2_vs_prompt1]).softmax(dim=0)
        results['image2']['prompt2'].append(result[0].item())
        results['image2']['prompt1'].append(result[1].item())

        text_correct = image1_vs_prompt1 > image1_vs_prompt2 and image2_vs_prompt2 > image2_vs_prompt1
        acc['text_correct'].append(text_correct.item())

        image_correct = image1_vs_prompt1 > image2_vs_prompt1 and image2_vs_prompt2 > image1_vs_prompt2
        acc['image_correct'].append(image_correct.item())

        acc['group_correct'].append(image_correct.item() and text_correct.item())
    return model_name, results, acc


def eval_finetuned_clip(model_weights, alpha, stream, model_name, pbar, results, acc):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    model.eval()

    # alpha = 0.2
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
    model_name = f'finetuned{stream}_ep20_alpha{alpha}_ViT-B32'
    model.load_state_dict(theta)

    logit_scale = model.logit_scale.exp().detach()
    for image1, image2, prompt1, prompt2 in pbar:
        batch_size = len(image1)
        with torch.no_grad():
            images = torch.stack([preprocess(image) for image in image1 + image2]).to(device)

            # Get features
            visual_features = model.encode_image(images)
            if isinstance(visual_features, tuple):
                visual_features = visual_features[0]

            text_features = model.encode_text(prompt1 + prompt2)
            if isinstance(text_features, tuple):
                text_features = text_features[0]

        visual_features = normalize(visual_features)
        text_features = normalize(text_features)

        image1_features = visual_features[:batch_size]
        image2_features = visual_features[batch_size:]

        prompt1_features = text_features[:batch_size]
        prompt2_features = text_features[batch_size:]

        batch_idx = 0
        # for batch_idx in range(batch_size):
        image1_vs_prompt1 = get_logits(logit_scale, image1_features[batch_idx], prompt1_features[batch_idx])
        image1_vs_prompt2 = get_logits(logit_scale, image1_features[batch_idx], prompt2_features[batch_idx])

        result = torch.stack([image1_vs_prompt1, image1_vs_prompt2]).softmax(dim=0)
        results['image1']['prompt1'].append(result[0].item())
        results['image1']['prompt2'].append(result[1].item())

        image2_vs_prompt2 = get_logits(logit_scale, image2_features[batch_idx], prompt2_features[batch_idx])
        image2_vs_prompt1 = get_logits(logit_scale, image2_features[batch_idx], prompt1_features[batch_idx])
        result = torch.stack([image2_vs_prompt2, image2_vs_prompt1]).softmax(dim=0)
        results['image2']['prompt2'].append(result[0].item())
        results['image2']['prompt1'].append(result[1].item())

        # result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]
        text_correct = image1_vs_prompt1 > image1_vs_prompt2 and image2_vs_prompt2 > image2_vs_prompt1
        acc['text_correct'].append(text_correct.item())

        # result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]
        image_correct = image1_vs_prompt1 > image2_vs_prompt1 and image2_vs_prompt2 > image1_vs_prompt2
        acc['image_correct'].append(image_correct.item())

        # image_correct(result) and text_correct(result)
        acc['group_correct'].append(image_correct.item() and text_correct.item())
    return model_name, results, acc


def eval_flava(pbar, results, acc):
    from transformers import FlavaProcessor, FlavaForPreTraining
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FlavaForPreTraining.from_pretrained("facebook/flava-full").to(device)
    preprocess = FlavaProcessor.from_pretrained("facebook/flava-full")
    for image1, image2, prompt1, prompt2 in pbar:
        with torch.no_grad():
            inputs = preprocess(
                text=prompt1 + prompt2,
                images=image1 + image2,
                return_tensors="pt",
                padding="max_length",
                max_length=77,
                return_codebook_pixels=True,
                return_image_mask=True,
                # Other things such as mlm_labels, itm_labels can be passed here. See docs
            ).to(device)
            inputs.bool_masked_pos.zero_()
            outputs = model(**inputs, input_ids_masked=inputs.input_ids)
            text_contrastive_logits = outputs.contrastive_logits_per_text

        image1_vs_prompts = text_contrastive_logits[0].softmax(dim=0)
        results['image1']['prompt1'].append(image1_vs_prompts[0].item())
        results['image1']['prompt2'].append(image1_vs_prompts[1].item())

        image2_vs_prompts = text_contrastive_logits[1].softmax(dim=0)
        results['image2']['prompt2'].append(image2_vs_prompts[1].item())
        results['image2']['prompt1'].append(image2_vs_prompts[0].item())

        image1_vs_prompt1 = text_contrastive_logits[0][0]
        image1_vs_prompt2 = text_contrastive_logits[0][1]

        image2_vs_prompt2 = text_contrastive_logits[1][1]
        image2_vs_prompt1 = text_contrastive_logits[1][0]

        text_correct = image1_vs_prompt1 > image1_vs_prompt2 and image2_vs_prompt2 > image2_vs_prompt1
        acc['text_correct'].append(text_correct.item())

        # result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]
        image_correct = image1_vs_prompt1 > image2_vs_prompt1 and image2_vs_prompt2 > image1_vs_prompt2
        acc['image_correct'].append(image_correct.item())

        # image_correct(result) and text_correct(result)
        acc['group_correct'].append(image_correct.item() and text_correct.item())
        # image_contrastive_logits = image_contrastive_logits.softmax(dim=1)
    return results, acc


def eval_visualbert(pbar, results, acc):
    #https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing
    # https://huggingface.co/docs/transformers/model_doc/visual_bert#transformers.VisualBertForMultipleChoice
    from transformers import AutoTokenizer, VisualBertForPreTraining
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre").to(device)

    """
    #https://github.com/huggingface/transformers/tree/main/examples/research_projects/visual_bert
    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
    image_preprocess = Preprocess(frcnn_cfg)
    
    bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    """
    for image1, image2, prompt1, prompt2 in pbar:
        pdb.set_trace()
        inputs = tokenizer(prompt1 + prompt2, return_tensors="pt").to(device)
        # with torch.no_grad():

def eval_bridgetower(pbar, results, acc):
    from transformers import BridgeTowerForImageAndTextRetrieval, BridgeTowerProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "BridgeTower/bridgetower-large-itm-mlm"
    preprocess = BridgeTowerProcessor.from_pretrained(model_id)
    model = BridgeTowerForImageAndTextRetrieval.from_pretrained(model_id).to(device)
    model.eval()

    for image1, image2, prompt1, prompt2 in pbar:
        with torch.no_grad():
            inputs = preprocess(images=image1 * 2 + image2 * 2,
                                text=prompt1 + prompt2 + prompt1 + prompt2,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=False,
                                max_length=40).to(device)

            outputs = model(**inputs).logits[:, 1].t()

        image1_vs_prompts = outputs[:2].softmax(dim=0)
        results['image1']['prompt1'].append(image1_vs_prompts[0].item())
        results['image1']['prompt2'].append(image1_vs_prompts[1].item())

        image2_vs_prompts = outputs[2:].softmax(dim=0)
        results['image2']['prompt2'].append(image2_vs_prompts[1].item())
        results['image2']['prompt1'].append(image2_vs_prompts[0].item())

        image1_vs_prompt1 = outputs[0]
        image1_vs_prompt2 = outputs[1]

        image2_vs_prompt2 = outputs[3]
        image2_vs_prompt1 = outputs[2]

        text_correct = image1_vs_prompt1 > image1_vs_prompt2 and image2_vs_prompt2 > image2_vs_prompt1
        acc['text_correct'].append(text_correct.item())

        image_correct = image1_vs_prompt1 > image2_vs_prompt1 and image2_vs_prompt2 > image1_vs_prompt2
        acc['image_correct'].append(image_correct.item())

        acc['group_correct'].append(image_correct.item() and text_correct.item())
    return results, acc


def eval_vilt(pbar, results, acc):
    # https://arxiv.org/pdf/2102.03334.pdf
    from transformers import ViltProcessor, ViltForImageAndTextRetrieval
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
    model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")
    model.to(device)

    for image1, image2, prompt1, prompt2 in pbar:
        with torch.no_grad():
            inputs = preprocess(images=image1 * 2,
                                text=prompt1 + prompt2,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=False,
                                max_length=40).to(device)
            outputs = model(**inputs)

            text_contrastive_logits = outputs.logits

            inputs = preprocess(image2 * 2,
                                prompt1 + prompt2,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=False,
                                max_length=40
                                ).to(device)
            outputs = model(**inputs)
        text_contrastive_logits = torch.cat([text_contrastive_logits.t(), outputs.logits.t()])
        image1_vs_prompts = text_contrastive_logits[0].softmax(dim=0)
        results['image1']['prompt1'].append(image1_vs_prompts[0].item())
        results['image1']['prompt2'].append(image1_vs_prompts[1].item())

        image2_vs_prompts = text_contrastive_logits[1].softmax(dim=0)
        results['image2']['prompt2'].append(image2_vs_prompts[1].item())
        results['image2']['prompt1'].append(image2_vs_prompts[0].item())

        image1_vs_prompt1 = text_contrastive_logits[0][0]
        image1_vs_prompt2 = text_contrastive_logits[0][1]

        image2_vs_prompt2 = text_contrastive_logits[1][1]
        image2_vs_prompt1 = text_contrastive_logits[1][0]

        text_correct = image1_vs_prompt1 > image1_vs_prompt2 and image2_vs_prompt2 > image2_vs_prompt1
        acc['text_correct'].append(text_correct.item())

        image_correct = image1_vs_prompt1 > image2_vs_prompt1 and image2_vs_prompt2 > image1_vs_prompt2
        acc['image_correct'].append(image_correct.item())

        # image_correct(result) and text_correct(result)
        acc['group_correct'].append(image_correct.item() and text_correct.item())
    return results, acc


def eval(model_name, annot_dir, root_image_dir, num_workers, batch_size, save_dir, model_weights, alpha, stream):
    """
    This will evaluate the confusion on non-sensible combinations versus sensible ones.
    Accuracy metrics are inspired by Winoground
        https://huggingface.co/datasets/facebook/winoground/blob/main/statistics/compute_statistics.py
    :param model:
    :param model_name:
    :param annot_dir:
    :param root_image_dir:
    :param num_workers:
    :param batch_size:
    :return:
    """
    logger.info("Building dataset...")
    for switch in [ 'composition', 'object']:
        logger.info(f"Running evaluation for when {switch} is switched...")
        dataset = COCOCompositionalDataset(annot_dir, root_image_dir, switch)
        dataloader = DataLoader(dataset, num_workers=num_workers, drop_last=False, batch_size=batch_size,
                                collate_fn=dataset.collate_fn)
        logger.info(f"Done building dataset with {len(dataset)} total samples and {len(dataloader)} batches.")
        pbar = tqdm(dataloader, total=len(dataloader), desc=f"Running for {model_name} on {switch} switch")

        # Create our dictionary of results.
        results = dict()
        columns = ['prompt1',
                   'prompt2']
        results['image1'] = dict()
        results['image2'] = dict()

        for key in results.keys():
            for col in columns:
                results[key][col] = list()

        columns = ['text_correct',
                   'image_correct',
                   'group_correct']

        acc = dict()
        for col in columns:
            acc[col] = list()


        if 'finetune' in model_name:
            model_name, results, acc = eval_finetuned_clip(model_weights, alpha, stream, model_name, pbar, results, acc)
        elif 'clip' in model_name:
            model_name, results, acc = eval_clip(model_name, pbar, results, acc)
        elif model_name == 'flava':
            results, acc = eval_flava(pbar, results, acc)
        elif model_name == 'vilt':
            results, acc = eval_vilt(pbar, results, acc)
        elif model_name == 'bridgetower':
            results, acc = eval_bridgetower(pbar, results, acc)
        else:
            raise NotImplementedError


        save_dir = os.path.join(save_dir, 'results')
        if not os.path.exists(save_dir):
            Path(save_dir).mkdir(parents=True)

        with open(os.path.join(save_dir, f'eval_compositional_switching_{switch}_understanding.json'), 'w') as f:
            json.dump({'rel_diff': results, 'acc': acc}, f)

        logger.info("-----------------------")
        logger.info(f"Softmax for {switch} switch:")
        for key, vals in results.items():
            for comparison, scores in vals.items():
                if len(scores) > 0:
                    results[key][comparison] = round(np.mean(scores), 4)
                    logger.info(f"Results for {key} for comparison {comparison}: {results[key][comparison]}")

        logger.info("-----------------------")
        logger.info(f"Accuracy for {switch} switch:")
        # for key, vals in acc.items():
        for comparison, scores in acc.items():
            if len(scores) > 0:
                acc[comparison] = round((np.sum(scores) / len(scores)), 4)
                logger.info(f"Results for {key} for comparison {comparison}: {acc[comparison]}")
        logger.info("-----------------------")


        with open(os.path.join(save_dir, f'eval_aggregated_compositional_switching_{switch}_understanding.json'), 'w') as f:
            json.dump({'rel_diff': results, 'acc': acc}, f)


@torch.jit.script
def normalize(x):
    return x / x.norm(dim=-1, keepdim=True, p=2)


@torch.jit.script
def get_logits(logit_scale, imf_i, text_f):
    # return logit_scale * imf_i @ text_f.t()
    return logit_scale * text_f @ imf_i.t()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs  a COCO based evaluation on different LLM+ models.')
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--checkpoint_pth', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--home_dir', type=str, default='/home/mschiappa/SRI/UnderstandingVisualTextModels')
    # Dataset Specific
    parser.add_argument('--annot_dir', type=str, default='data/coco')
    parser.add_argument('--root_image_dir', default='/media/mschiappa/Elements/coco/val2014', type=str)
    # parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_weights', type=str,
                        default='/home/schiappa/SRI/UnderstandingVisualTextModels/output/ViT-B32/finetune/lr0.001_wd0.0001/run1/clip_pt_ep10.pth')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--stream', type=str, default='vt')

    args = parser.parse_args()

    # Build our save directory for results, must pass the top-level directory results are stored in
    save_dir = os.path.join(args.save_dir, 'compositional_text_eval', args.model)
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

    eval(args.model, args.annot_dir, args.root_image_dir, args.num_workers, 1, save_dir, args.model_weights, args.alpha,
         args.stream)