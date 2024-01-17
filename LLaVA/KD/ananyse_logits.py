import argparse
import sys
import os
import torch
from tqdm import tqdm
sys.path.append("../LLaVA")
sys.path.append("../LLaVA/llava")
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.train.train import LazySupervisedDataset, make_supervised_data_module
from torch.utils.data import DataLoader
from transformers.trainer import get_parameter_names, ALL_LAYERNORM_LAYERS
import torch.nn as nn
import traceback
import torch.nn.functional as F


from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    logger,
)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,args.load_8bit, args.load_4bit, device=args.device)
    model.eval()

    t_model_name = get_model_name_from_path(args.t_model_path)
    t_tokenizer, t_model, t_image_processor, t_context_len = load_pretrained_model(args.t_model_path, args.model_base, t_model_name,args.load_8bit, args.load_4bit, device=args.device)
    t_model.eval()

    vision_tower = model.get_vision_tower()
    args.image_processor = vision_tower.image_processor
    args.is_multimodal = True

    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,data_path=args.data_path, data_args=args)


    batch_size = 4

    t_all_logits_mean = []
    all_logits_mean = []

    for i in tqdm(train_dataset):
        labels = i["labels"].to(model.device) 
        full_input_ids = i["input_ids"].unsqueeze(dim=0).to(model.device)
        image_tensor_ = i["images"].half().unsqueeze(dim=0).to(model.device)
        has_img = i["has_img"]

        # 找到所有大于-100的索引
        big_mask_indices = (labels > -100).nonzero(as_tuple=False).squeeze()[:2000].to(model.device)
        # big_mask_indices = big_mask_indices[torch.randperm(big_mask_indices.shape[0])]
        max_index = torch.max(big_mask_indices)
        big_mask = (torch.arange(labels.shape[0]).unsqueeze(0).to(model.device) <= (big_mask_indices-1).unsqueeze(1))

        in_batch_time = (len(big_mask_indices) + batch_size - 1) // batch_size

        # print(labels)
        # print(full_input_ids)
        # print(big_mask_indices)

        for i in range(in_batch_time):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(big_mask_indices))
            mask = big_mask[start_idx:end_idx]
            mask_indices = big_mask_indices[start_idx:end_idx]
            input_ids = full_input_ids.repeat(mask.shape[0], 1)
            image_tensor = image_tensor_.repeat(mask.shape[0], 1, 1, 1)
            position_ids = torch.arange(0,input_ids.shape[1]).unsqueeze(dim=0).repeat(mask.shape[0], 1)

            with torch.no_grad():
                t_output = t_model(
                    input_ids = input_ids, 
                    attention_mask = mask,
                    position_ids = position_ids,
                    use_cache = True,
                    output_attentions = False,
                    output_hidden_states = False,
                    images = image_tensor,
                    return_dict = True,
                )

                bs,length = t_output.logits.shape[0], t_output.logits.shape[1]

                if has_img:
                    index = (mask_indices+574).to(model.device).unsqueeze(1).repeat(1,32000).reshape(bs,1,32000)
                else:
                    index = (mask_indices-1).to(model.device).unsqueeze(1).repeat(1,32000).reshape(bs,1,32000)

                t_result = torch.gather(t_output.logits, 1, index).squeeze(1).cpu()

                output = model(
                        input_ids = input_ids, 
                        attention_mask = mask,
                        position_ids = position_ids,
                        use_cache = True,
                        output_attentions = False,
                        output_hidden_states = False,
                        images = image_tensor,
                        return_dict = True,
                    )

                result = torch.gather(output.logits, 1, index).squeeze(1).cpu()

                # analyse
                t_logits_mean = torch.mean(t_result, dim=-1)
                logits_mean = torch.mean(result, dim=-1)
                
                with open("../save/ana.txt", "a") as f:
                    for i in range(t_logits_mean.size(0)):
                        f.write(str(float(t_logits_mean[i]))+" "+str(float(logits_mean[i]))+"\n")

        
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_model-path", type=str, default="../13B/")
    parser.add_argument("--model-path", type=str, default="../ft/")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="../")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", type=bool, default=False)
    parser.add_argument("--debug", action="store_true")
    
    parser.add_argument("--data-path", type=str, default="../llava_v1_5_mix665k.json")
    parser.add_argument("--image-folder", type=str, default="../")
    parser.add_argument("--mm_use_im_start_end", type=bool, default=False)
    parser.add_argument("--mm_use_im_patch_token", type=bool, default=False)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")

    args = parser.parse_args()
    main(args)