#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# Here you can see the first stage of CoRe method

import argparse
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
import yaml
import glob

import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
from torch.nn import HingeEmbeddingLoss 
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from sklearn.metrics import mean_squared_error as mse
from resnet_pytorch import ResNet 

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import wandb

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import Attention as attn
from diffusers.utils.torch_utils import is_compiled_module
from nb_utils import eval_sets_core
from nb_utils.clip_eval import ExpEvaluator
from nb_utils.configs import _LOAD_IMAGE_BACKEND


if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)


def save_model_card(repo_id: str, images: list = None, base_model: str = None, repo_folder: str = None):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"
    model_description = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. \n
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "textual_inversion",
        "diffusers-training",
        "CoRe"
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))



def evaluate_validation(args, evaluator, images, prompt):
    paths = sorted(glob.glob(os.path.join(args.test_data_dir, '*')))
    train_images = [_LOAD_IMAGE_BACKEND(path) for path in paths]
    train_images_features, _ = evaluator._get_image_features(train_images, args.resolution)
    images_features, _ = evaluator._get_image_features(images, args.resolution)

    image_similarities, _ = evaluator._calc_similarity(train_images_features, images_features)
    clean_label = (
        prompt
        .replace('{0} {1}'.format(args.placeholder_token, args.class_name), '{0}')
        .replace('{0}'.format(args.placeholder_token), '{0}')
    )
    empty_label = clean_label.replace('{0} ', '').replace(' {0}', '')
    empty_label_features = evaluator.evaluator.get_text_features(empty_label)
    text_similarities, _ = evaluator._calc_similarity(empty_label_features, images_features)
    return image_similarities, text_similarities


def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch, evaluator, original_is=None, original_ts=None):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompts:"
        f" {validation_prompt_set}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    image_similarities = []
    text_similarities = []
    captions = []
    for prompt in validation_prompt_set:
        imgs = pipeline(
            prompt.format(f'{args.placeholder_token}'), #  {args.class_name}
            num_images_per_prompt=args.num_validation_images,
            num_inference_steps=50, 
            generator=generator
        ).images
        
        images += imgs
        captions += [prompt] * args.num_validation_images
        ims, ts = evaluate_validation(args, evaluator, [np.asarray(i) for i in imgs], prompt)
        image_similarities.append(ims)
        text_similarities.append(ts)

    # for _ in range(args.num_validation_images):
    #     if torch.backends.mps.is_available():
    #         autocast_ctx = nullcontext()
    #     else:
    #         autocast_ctx = torch.autocast(accelerator.device.type)

    #     with autocast_ctx:
    #         image = pipeline(validation_prompt, num_inference_steps=25, generator=generator).images[0]
    #     images.append(image)

    for tracker in accelerator.trackers:
        # if tracker.name == "tensorboard":
        #     np_images = np.stack([np.asarray(img) for img in images])
        #     tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        # if tracker.name == "wandb":
        #     tracker.log(
        #         {
        #             "validation": [
        #                 wandb.Image(image, caption=f"{i}: {validation_prompt}") for i, image in enumerate(images)
        #             ]
        #         }
        #     )
            
        if original_is and original_ts:
            val_is_delta = np.mean(image_similarities)
            val_ts_delta = np.mean(text_similarities) / original_ts
            tracker.log({
                "val_is_delta": val_is_delta,
                "val_ts_delta": val_ts_delta,
                "val_delta": 2 / (1 / val_is_delta + 1 / val_ts_delta),
                "validation": [wandb.Image(images[i].resize((128, 128)), caption=captions[i]) for i in range(len(images))],
                "val_is": np.mean(image_similarities),
                "val_ts": np.mean(text_similarities),
                "val_metrics": 2 / (1 / np.mean(image_similarities) + 1 / np.mean(text_similarities)),
            })        

    del pipeline
    torch.cuda.empty_cache()
    return images, np.mean(image_similarities), np.mean(text_similarities)


def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--test_data_dir", type=str, default=None, required=True, help="A folder containing the testing data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word."
    )
    # parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--concept_property", type=str, default="object", help="Choose between 'object' and 'live'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--class_name",
        type=str,
        default=None,
        help="The name of the prompt class"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        required=True,
        help="The name of the experiment",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=300,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--train_text_encoder",
        action='store_true', 
        help="Flag for text encoder training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and Nvidia Ampere GPU or Intel Gen 4 Xeon (and later) ."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # parser.add_argument(
    #     "--validation_prompt",
    #     type=str,
    #     default=None,
    #     help="A prompt that is used during validation to verify that the model is learning.",
    # )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=5,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--add_emb_reg",
        type=lambda x: x.lower() in ("true", "1"),  # Accepts 'True', '1', 'False', '0'
        default=False,
        help="Add embedding regularization."
    )
    parser.add_argument(
        "--add_attn_reg",
        type=lambda x: x.lower() in ("true", "1"),  # Accepts 'True', '1', 'False', '0'
        default=False,
        help="Add attention regularization."
    )
    parser.add_argument(
        "--add_pairwise_reg",
        type=lambda x: x.lower() in ("true", "1"),  # Accepts 'True', '1', 'False', '0'
        default=False,
        help="Add pairwise regularization."
    )
    parser.add_argument(
        "--add_hinge_reg",
        type=lambda x: x.lower() in ("true", "1"),  # Accepts 'True', '1', 'False', '0'
        default=False,
        help="Add Hinge regularization."
    )
    parser.add_argument(
        "--lower_hinge_loss_bound",
        type=float,  
        default=0.5,
        help="Lower bound for Hinge loss."
    )
    parser.add_argument(
        "--upper_hinge_loss_bound",
        type=float,
        default=0.6,
        help="Upper bound for Hinge loss."
    )
    parser.add_argument(
        "--lambda_emb",
        type=float,
        default=1.5e-4,
        help="Lambda coefficient for embedding loss."
    )
    parser.add_argument(
        "--lambda_attn",
        type=float,
        default=0.05,
        help="Lambda coefficient for attention loss."
    )
    parser.add_argument(
        "--lambda_pairwise",
        type=float,
        default=0.015,
        help="Lambda coefficient for pairwise loss."
    )
    parser.add_argument(
        "--lambda_hinge",
        type=float,
        default=1,
        help="Lambda coefficient for Hinge loss."
    )


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


validation_prompt_set = [
    'a {} on a snowy mountaintop, partially buried under fresh, powdery snow', # background
    'a {} on a tranquil lake in a rowboat, with mist rising at dawn', # background
    'a {} captured in the soft light of an impressionist painting', # style
    'a traditional Chinese painting of a {}', # style
    'a black {} seen from the top', # color change
]


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        superclass="dog",
        # learnable_property="object",  # [object, style]
        concept_property="object",  # [object, live]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        # self.learnable_property = learnable_property
        self.concept_property = concept_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.superclass = superclass
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        # self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.templates = eval_sets_core.live_set if concept_property == "live" else eval_sets_core.object_set
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        superclass = self.superclass
        p = random.choice(self.templates)
        text1 = p.format(superclass)
        text2 = p.format(placeholder_string) 
        instance_text = 'a photo of a {0}'.format(placeholder_string)
        
        text2_words = text2.split()
        if placeholder_string not in text2_words:
            print(f"Placeholder string '{placeholder_string}' not found in text2_words: {text2_words}")
            raise ValueError(f"'{placeholder_string}' is not in list")

        word_id = text2_words.index(placeholder_string)
        
        if text2_words[word_id] != placeholder_string:
            print("Error: text2_words[word_id] is not equal to placeholder_string")
            print(f"word_id = {word_id}, text2[word_id] = {text2_words[word_id]}")


        example["input_ids"] = self.tokenizer(
            instance_text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        # print(example["input_ids"])

        example["text1"] = self.tokenizer(
            text1,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        # print(example["text1"])

        example["text2"] = self.tokenizer(
            text2,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        # print(example["text2"])

        # exapmle["word_id"] = word_id

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example
    

class SaveOutput:
    def __init__(self, register_outputs=True, register_inputs=False):
        self.outputs = {}
        self.inputs = {}
        self.counter = {}
        self.register_outputs = register_outputs
        self.register_inputs = register_inputs

    def __call__(self, module, module_in, module_out):
        if not hasattr(module, 'module_name'):
            raise AttributeError('All modules should have name attr')
        if self.register_outputs:
            self.outputs[module.module_name] = module_out
        if self.register_inputs:
            self.inputs[module.module_name] = module_in

    def clear(self):
        self.outputs = {}
        self.inputs = {}
        self.counter = 0


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    # logging_dir = os.path.join(args.output_dir, args.logging_dir)
    logging_dir = Path(args.output_dir, args.logging_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(logging_dir, "hparams.yml"), "w") as outfile:
              yaml.dump(vars(args), outfile)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id


    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
        
    # Add the placeholder token in tokenizer
    placeholder_tokens = [args.placeholder_token]
    suprerclass_tokens = [args.class_name]

    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    # As I understand token_embeds are previous embeddings

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    for name, layer in unet.named_modules():
        layer.module_name = name
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    
    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        superclass=args.class_name,
        size=args.resolution,
        placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))),
        repeats=args.repeats,
        concept_property = args.concept_property,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    
    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    text_encoder.train()
    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("CoRe", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    evaluator = ExpEvaluator(accelerator.device)
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    def get_cross_attnention_maps(save_output):
        attention_probs_list = []
        for name, layer in unet.named_modules():
            if name.endswith('attn2'):
                heads = layer.heads
                name_q = name + '.to_q'
                name_k = name + '.to_k'
                query = save_output.outputs[name_q]
                key = save_output.outputs[name_k]
                query = attn.head_to_batch_dim(layer, query)
                key = attn.head_to_batch_dim(layer, key)

                # get cross attention maps
                attention_probs = attn.get_attention_scores(layer, query, key)

                # mean across heads
                batch_size = attention_probs.shape[0] // heads
                seq_len = attention_probs.shape[1]
                dim = attention_probs.shape[-1]
        
                attention_probs = attention_probs.view(batch_size, heads, seq_len, dim)
                attention_probs = attention_probs.mean(dim=1)

                sqrt_seq_len = int(math.sqrt(seq_len))
                attention_probs = attention_probs.view(batch_size, sqrt_seq_len, sqrt_seq_len, dim)                        
                attention_probs_list.append(attention_probs)
        
        return attention_probs_list


    def interpolate_cross_attention_maps(attention_probs_list):
        min_seq_len = min([ca.shape[1] for ca in attention_probs_list])
        target_size = (min_seq_len, min_seq_len)
        interp_attention_maps = []
        for ca_map in attention_probs_list:
            interp_ca_map = F.interpolate(ca_map.permute(0, 3, 1, 2), size=target_size, mode='bilinear')
            interp_ca_map = interp_ca_map.permute(0, 2, 3, 1)
            interp_attention_maps.append(interp_ca_map)
        interp_attention_maps = torch.stack(interp_attention_maps)
        return interp_attention_maps


    # def hinge_loss(cos_sims, lower_bound=0.5, upper_bound=0.6):
    #     loss_lower = torch.max(torch.zeros_like(cos_sims), lower_bound - cos_sims)
    #     loss_upper = torch.max(torch.zeros_like(cos_sims), cos_sims - upper_bound)
    #     return loss_lower + loss_upper


    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone() #.cpu()

    loss_emb_list = []
    loss_attn_list = []
    loss_pairwise_list = []
    cls_concept_cos_sims = []
    attn_maps1 = []
    attn_maps2 = []
    pairwise_cos_sim_data = []

    print(type(text_encoder))
    print(text_encoder.device)
    print(type(tokenizer))
    print(type(unet))
    print(unet.device)
    print(type(vae))
    print(vae.device)    
    print(type(accelerator))
    print(accelerator.device)
    print(weight_dtype)
    print(type(evaluator))
    print(evaluator.device)

    _, is0, ts0 = log_validation(
        text_encoder,
        tokenizer,
        unet,
        vae,
        args,
        accelerator,
        weight_dtype,
        0,
        evaluator,
    )

        
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach() #.cpu()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embeddings for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
                encoder_hidden_states1 = text_encoder(batch["text1"])[0].clone().to(dtype=weight_dtype)
                encoder_hidden_states2 = text_encoder(batch["text2"])[0].clone().to(dtype=weight_dtype)
                # encoder_hidden_states2: (1, 77, 1024) = (batch_size, sequence_length, feature_dim)

                # word_id = torch.argmax(batch["text2"][0]).item() 

                # # Count cosine similarities between superclass tokens and placeholder tokens
                # # For regularization we count cos_sim between all tokens  exept them  
                # cls_concept_cos_sim = F.cosine_similarity(
                #     encoder_hidden_states1[:, word_id, :], 
                #     encoder_hidden_states2[:, word_id, :], 
                #     dim=-1
                # ) 
                # cls_concept_cos_sims.append(cls_concept_cos_sim.detach().cpu().numpy())

                # # Count cosine similarities between tokens in superclass and placeholder embeddings
                # # Get prompts 
                # text1 = tokenizer.batch_decode(batch["text1"], skip_special_tokens=True)
                # text2 = tokenizer.batch_decode(batch["text2"], skip_special_tokens=True)

                # # Pairwise cosine similarities within embeddings
                # # superclass embedding
                # cosine_matrix_cls = F.cosine_similarity(
                #     encoder_hidden_states1.unsqueeze(2), 
                #     encoder_hidden_states1.unsqueeze(1), 
                #     dim=-1
                # ).cpu().detach().numpy()
    
                # # placeholder embedding
                # cosine_matrix_plh = F.cosine_similarity(
                #     encoder_hidden_states2.unsqueeze(2), 
                #     encoder_hidden_states2.unsqueeze(1), 
                #     dim=-1
                # ).cpu().detach().numpy()
                # # tokens_cos_sims_placeholder.append({cosine_matrix2, text2})
                # # print('\ncls == plh', (cosine_matrix_cls == cosine_matrix_plh).all())

                # diff_matrix = np.abs(cosine_matrix_cls - cosine_matrix_plh)
                # loss_pairwise = torch.sum(torch.from_numpy(diff_matrix))
                
                
                # clean_encoder_hidden_states1 = torch.cat((encoder_hidden_states1[:, :word_id, :], encoder_hidden_states1[:, word_id+1:, :]), dim=1)
                # clean_encoder_hidden_states2 = torch.cat((encoder_hidden_states2[:, :word_id, :], encoder_hidden_states2[:, word_id+1:, :]), dim=1)
                # # encoder_hidden_states2[:, word_id, :] = 0

                # cos_sim = F.cosine_similarity(clean_encoder_hidden_states1, clean_encoder_hidden_states2, dim=-1) 
                # loss_emb = torch.sum(1 - cos_sim) / clean_encoder_hidden_states1.shape[1] 
                # # (0.8 - cos_sim) по модулю

                # save_output = SaveOutput()
                # hook_handles = []
                # for name, layer in unet.named_modules():
                #     if 'attn2.to_q' in name or 'attn2.to_k' in name:
                #         handle = layer.register_forward_hook(save_output)
                #         hook_handles.append(handle)
                
                # model_pred_1 = unet(
                #     noisy_latents, 
                #     timesteps, 
                #     encoder_hidden_states1
                # )
                
                # attention_probs_list1 = get_cross_attnention_maps(save_output)
                # interp_attention_maps1 = interpolate_cross_attention_maps(attention_probs_list1)
                # mean_attn_map1 = interp_attention_maps1.mean(dim=0)
                # attn_maps1.append(mean_attn_map1.cpu().detach().numpy())
                # mean_attn_map1[:, :, :, word_id] = 0

                # for handle in hook_handles:
                #     handle.remove()


                # save_output = SaveOutput()
                # hook_handles = []
                # for name, layer in unet.named_modules():
                #     if 'attn2.to_q' in name or 'attn2.to_k' in name:
                #         handle = layer.register_forward_hook(save_output)
                #         hook_handles.append(handle)
                
                # model_pred_2 = unet(
                #     noisy_latents, 
                #     timesteps, 
                #     encoder_hidden_states2
                # )

                # attention_probs_list2 = get_cross_attnention_maps(save_output)
                # interp_attention_maps2 = interpolate_cross_attention_maps(attention_probs_list2)
                # mean_attn_map2 = interp_attention_maps2.mean(dim=0)
                # attn_maps2.append(mean_attn_map2.cpu().detach().numpy())
                # mean_attn_map2[:, :, :, word_id] = 0

                # for handle in hook_handles:
                #     handle.remove()

                # n = mean_attn_map1.shape[-1]
                # loss_attn = F.mse_loss(mean_attn_map1, mean_attn_map2, reduction="sum") 
                # loss_attn /= (n - 1)

                # pairwise_cos_sim_data.append({
                #     "epoch": epoch,
                #     "step": step,
                #     "text1": text1,
                #     "text2": text2,
                #     "cosine_matrix_cls": cosine_matrix_cls,
                #     "cosine_matrix_plh": cosine_matrix_plh,
                #     "attn_maps1": attn_maps1,
                #     "attn_maps2": attn_maps2
                # })

                # DONE: Move to args 
                # lambda_emb = 1.5e-4
                # lambda_attn = 0.05
                # lambda_pairwise = 0.015
                # lambda_hinge = 1
                
                word_ids = [torch.argmax(batch["text2"][i]).item() for i in range(bsz)]
                # print('word_ids: ', word_ids)
                
                cos_sims = []

                for i in range(bsz):
                    word_id = word_ids[i]
            
                    # Count cosine similarities between superclass tokens and placeholder tokens
                    # For regularization we count cos_sim between all tokens exept them  
                    cls_concept_cos_sim = F.cosine_similarity(
                        encoder_hidden_states1[i, word_id, :], 
                        encoder_hidden_states2[i, word_id, :], 
                        dim=-1
                    ) 
                    # print('cls_concept_cos_sim: ', cls_concept_cos_sim.detach().cpu().numpy())
                    cos_sims.append(cls_concept_cos_sim.detach().cpu().numpy())
                cls_concept_cos_sims.append(np.array(cos_sims))
                
                # print('cls_concept_cos_sims: ', cls_concept_cos_sims)
                # print('cls_concept_cos_sims.shape: ', np.array(cls_concept_cos_sims, dtype=object).shape)
                # print('cls_concept_cos_sims.shape[0]: ', np.array(cls_concept_cos_sims, dtype=object).shape[0])

                # Count cosine similarities between tokens in superclass and placeholder embeddings
                # Get prompts 
                text1 = tokenizer.batch_decode(batch["text1"], skip_special_tokens=True)
                # print('text1: ', text1)
                text2 = tokenizer.batch_decode(batch["text2"], skip_special_tokens=True)
                # print('text2: ', text2)

                # Pairwise cosine similarities within embeddings
                # superclass embedding
                cosine_matrix_cls = F.cosine_similarity(
                    encoder_hidden_states1.unsqueeze(2), 
                    encoder_hidden_states1.unsqueeze(1), 
                    dim=-1
                ).cpu().detach().numpy()
    
                # placeholder embedding
                cosine_matrix_plh = F.cosine_similarity(
                    encoder_hidden_states2.unsqueeze(2), 
                    encoder_hidden_states2.unsqueeze(1), 
                    dim=-1
                ).cpu().detach().numpy()
                # tokens_cos_sims_placeholder.append({cosine_matrix2, text2})
                # print('\ncls == plh', (cosine_matrix_cls == cosine_matrix_plh).all())

                diff_matrix = np.abs(cosine_matrix_cls - cosine_matrix_plh)
                loss_pairwise = torch.sum(torch.from_numpy(diff_matrix)) 
                loss_pairwise_list.append((loss_pairwise * args.lambda_pairwise).detach().cpu().numpy())
                
                # print('cosine_matrix_cls.shape: ', cosine_matrix_cls.shape)
                # print('cosine_matrix_plh.shape: ', cosine_matrix_plh.shape)

                pairwise_cos_sim_data.append({
                    "epoch": epoch,
                    "step": step,
                    "text1": text1,
                    "text2": text2,
                    "cosine_matrix_cls": cosine_matrix_cls,
                    "cosine_matrix_plh": cosine_matrix_plh,
                    # "attn_maps1": attn_maps1,
                    # "attn_maps2": attn_maps2
                })

               
                # Context embedding loss
                loss_emb = 0
                for i in range(bsz):
                    word_id = word_ids[i]
                    # print('word_id: ', word_id)
                    
                    # clean_encoder_hidden_states1 = torch.cat((
                    #     encoder_hidden_states1[i, :word_id, :], 
                    #     encoder_hidden_states1[i, word_id+1:, :]
                    # ), dim=1)
                    
                    # clean_encoder_hidden_states2 = torch.cat((
                    #     encoder_hidden_states2[i, :word_id, :], 
                    #     encoder_hidden_states2[i, word_id+1:, :]
                    # ), dim=1)

                    clean_encoder_hidden_states1 = encoder_hidden_states1.clone()
                    clean_encoder_hidden_states2 = encoder_hidden_states2.clone()

                    clean_encoder_hidden_states1[i, word_id, :] = 0
                    clean_encoder_hidden_states2[i, word_id, :] = 0

                cos_sim = F.cosine_similarity(
                    clean_encoder_hidden_states1, 
                    clean_encoder_hidden_states2, 
                    dim=-1
                ) 
                # print('cos_sim.shape: ', cos_sim.shape)
                # print('cos_sim: ', cos_sim)

                loss_emb += torch.sum(1 - cos_sim) / (clean_encoder_hidden_states1.shape[1]-1)
                # loss_emb = torch.sum(1 - cos_sim) / clean_encoder_hidden_states1.shape[1] 
                # loss_emb /= bsz
                loss_emb_list.append((loss_emb * args.lambda_emb).detach().cpu().numpy())
                
                
                # Context attention loss
                loss_attn = 0

                # Generate attention for encoder_hidden_states1 
                save_output = SaveOutput()
                hook_handles = []
                for name, layer in unet.named_modules():
                    if 'attn2.to_q' in name or 'attn2.to_k' in name:
                        handle = layer.register_forward_hook(save_output)
                        hook_handles.append(handle)
                
                model_pred_1 = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states1
                )
                
                attention_probs_list1 = get_cross_attnention_maps(save_output)
                interp_attention_maps1 = interpolate_cross_attention_maps(attention_probs_list1)
                mean_attn_map1 = interp_attention_maps1.mean(dim=0)
                attn_maps1.append(mean_attn_map1.cpu().detach().numpy())
                
                for i in range(bsz):
                    word_id = word_ids[i]
                    mean_attn_map1[i, :, :, word_id] = 0
    
                for handle in hook_handles:
                    handle.remove()

                # Generate attention for encoder_hidden_states1 
                save_output = SaveOutput()
                hook_handles = []
                for name, layer in unet.named_modules():
                    if 'attn2.to_q' in name or 'attn2.to_k' in name:
                        handle = layer.register_forward_hook(save_output)
                        hook_handles.append(handle)
                
                model_pred_2 = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states2
                )

                attention_probs_list2 = get_cross_attnention_maps(save_output)
                interp_attention_maps2 = interpolate_cross_attention_maps(attention_probs_list2)
                mean_attn_map2 = interp_attention_maps2.mean(dim=0)
                attn_maps2.append(mean_attn_map2.cpu().detach().numpy())

                for i in range(bsz):
                    word_id = word_ids[i]
                    mean_attn_map2[i, :, :, word_id] = 0
    
                for handle in hook_handles:
                    handle.remove()

                n = mean_attn_map1.shape[-1]
                loss_attn += F.mse_loss(mean_attn_map1, mean_attn_map2, reduction="sum") / (n - 1)
                # loss_attn /= bsz
                loss_attn_list.append((loss_attn * args.lambda_attn).detach().cpu().numpy())

                
                # Predict the noise residual
                model_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


                # From dtype=object to dtype=float32
                # cls_concept_cos_sims = np.array([arr for arr in cls_concept_cos_sims])
                # From np.array to torch.tensor
                cos_sims = torch.tensor(np.array(cos_sims), dtype=torch.float32)

                # DONE: Move to args
                # lower_hinge_loss_bound = 0.7
                # upper_hinge_loss_bound = 0.8

                hinge_loss_lower = torch.max(
                    torch.zeros_like(cos_sims), 
                    args.lower_hinge_loss_bound - cos_sims
                )
                
                hinge_loss_upper = torch.max(
                    torch.zeros_like(cos_sims), 
                    cos_sims - args.upper_hinge_loss_bound
                )

                loss_hinge = torch.mean(hinge_loss_lower + hinge_loss_upper)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # hinge = HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean') 
                # loss_hinge = F.hinge_embedding_loss(cls_concept_cos_sims, [0.5, 0.6], reduction='mean')
                # hinge_loss2 = F.hinge_embedding_loss(
                #     cls_concept_cos_sims, 
                #     torch.tensor(0.5, dtype=torch.float32), 
                #     # reduce=False, 
                #     reduction='mean'
                # ) + F.hinge_embedding_loss(
                #     cls_concept_cos_sims, 
                #     torch.tensor(0.6, dtype=torch.float32), 
                #     # reduce=False, 
                #     reduction='mean'
                # )

                # Count loss with(out) any regularization
                if args.add_emb_reg:
                    loss += args.lambda_emb * loss_emb 
                if args.add_attn_reg:
                    loss += args.lambda_attn * loss_attn
                if args.add_pairwise_reg:
                    loss += args.lambda_pairwise * loss_pairwise
                if args.add_hinge_reg:
                    loss += args.lambda_hinge * loss_hinge
                    

                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

                v_prev = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[-1]#.cpu()
            
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                v_curr = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[-1] #.cpu()
                v_curr = v_curr / torch.norm(v_curr, dim=-1, keepdim=True) * torch.norm(v_prev, dim=-1, keepdim=True)
                # check dim
                # count cos sim

                if step >= 240 and step<=360:
                    orig_embeds_params[-1] = v_curr 

                # Let's make sure we don't update any embedding weights besides the newly added token
                # index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                # index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False
                # index_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = True

                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates] #.cuda() # v_curr
    

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                images = []
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    weight_name = (
                        f"learned_embeds.bin"
                        if args.no_safe_serialization
                        else f"learned_embeds.safetensors"
                    )
                    
                    save_path = Path(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                    save_path.mkdir(parents=True, exist_ok=True)
                    print(f'save path dir: {save_path}')
                    save_path = os.path.join(save_path, weight_name)
                    print(f'save path full: {save_path}')
                    save_progress(
                        text_encoder,
                        placeholder_token_ids,
                        accelerator,
                        args,
                        save_path,
                        safe_serialization=not args.no_safe_serialization,
                    )

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # Here we bring a random promp for validation from the orig article
                        # if args.concept_property == 'live':
                        #     validation_prompt = random.choice(eval_sets_core.live_set)
                        # else:
                        #     validation_prompt = random.choice(eval_sets_core.object_set)
                        # validation_prompt = validation_prompt.format(args.placeholder_token)
    
                        # if validation_prompt is not None and global_step % args.validation_steps == 0:
                        #     images = log_validation(
                        #         text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, global_step, validation_prompt
                        #     )

                        images, _, _ = log_validation(
                            text_encoder,
                            tokenizer,
                            unet,
                            vae,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            evaluator,
                            is0, ts0,
                        )

            logs = {
                "loss_hinge": loss_hinge.detach().item(), 
                "loss_emb": loss_emb.detach().item(), 
                "loss_attn": loss_attn.detach().item(), 
                "loss_pairwise": loss_pairwise.detach().item(),
                "loss_total": loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0], 
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    save_path = os.path.join(args.output_dir, "text_encoder.bin")
    torch.save(text_encoder.state_dict(), save_path)
    print(f"Text encoder saved to {save_path}")      

    save_path = os.path.join(args.output_dir, "cls_concept_cos_sims.npy")
    np.save(save_path, np.array(cls_concept_cos_sims, dtype=object))
    print(f"Cosine similarities between superclass tokens and placeholder tokens saved to {save_path}")

    save_path = os.path.join(args.output_dir, "pairwise_cos_sim_data.npy")
    np.save(save_path, np.array(pairwise_cos_sim_data, dtype=object))
    print(f"Pairwise cosine similarities between tokens in superclass and placeholder embeddings saved to {save_path}")

    save_path = os.path.join(args.output_dir, "loss_emb.npy")
    np.save(save_path, np.array(loss_emb_list))
    print(f"Emb losses saved to {save_path}")   

    save_path = os.path.join(args.output_dir, "loss_attn.npy")
    np.save(save_path, np.array(loss_attn_list))
    print(f"Attn losses saved to {save_path}")

    save_path = os.path.join(args.output_dir, "loss_pairwise.npy")
    np.save(save_path, np.array(loss_pairwise_list))
    print(f"Attn losses saved to {save_path}")



    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and not args.save_as_full_pipeline:
            logger.warning("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = args.save_as_full_pipeline
        if save_full_model:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(args.output_dir)
        # Save the newly trained embeddings
        weight_name = "learned_embeds.bin" if args.no_safe_serialization else "learned_embeds.safetensors"
        save_path = os.path.join(args.output_dir, weight_name)
        save_progress(
            text_encoder,
            placeholder_token_ids,
            accelerator,
            args,
            save_path,
            safe_serialization=not args.no_safe_serialization,
        )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
