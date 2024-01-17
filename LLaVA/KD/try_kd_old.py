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
from torch.utils.data.distributed import DistributedSampler

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse  
# 负责创建 args.local_rank 变量，并接受 torch.distributed.launch 注入的值
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

# 每个进程根据自己的local_rank设置应该使用的GPU
torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)

# 初始化分布式环境，主要用来帮助进程间通信
torch.distributed.init_process_group(backend='nccl')


def main(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,args.load_8bit, args.load_4bit, device=args.device)
    model.train()

    t_model_name = get_model_name_from_path(args.t_model_path)
    t_tokenizer, t_model, t_image_processor, t_context_len = load_pretrained_model(args.t_model_path, args.model_base, model_name,args.load_8bit, args.load_4bit, device=args.device)
    t_model.eval()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    t_model = torch.nn.parallel.DistributedDataParallel(t_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    vision_tower = model.get_vision_tower()
    args.image_processor = vision_tower.image_processor
    args.is_multimodal = True

    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,data_path=args.data_path, data_args=args)
    
    op = torch.optim.SGD(model.parameters(), lr=1e-4)
    loss_func = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean').to(device)

    epoch_min = 0
    batch_size = 32

    for i in tqdm(train_dataset):
        epoch_min = epoch_min + 1
        labels = i["labels"] 
        full_input_ids = i["input_ids"].unsqueeze(dim=0)
        image_tensor_ = i["images"].half().unsqueeze(dim=0)

        # 找到所有大于-100的索引
        big_mask_indices = (labels > -100).nonzero(as_tuple=False).squeeze()
        mask_ = big_mask_indices <= 2000
        big_mask_indices = big_mask_indices[mask_]
        max_index = torch.max(big_mask_indices)
        big_mask = (torch.arange(labels.shape[0]).unsqueeze(0) <= big_mask_indices.unsqueeze(1))

        in_batch_time = (len(big_mask_indices) + batch_size - 1) // batch_size

        op.zero_grad()
        for i in range(in_batch_time):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(big_mask_indices))
            mask = big_mask[start_idx:end_idx]
            mask_indices = big_mask_indices[start_idx:end_idx].to(device)

            input_ids = full_input_ids.repeat(mask.shape[0], 1)
            image_tensor = image_tensor_.repeat(mask.shape[0], 1, 1, 1)
            position_ids = torch.arange(0,input_ids.shape[1]).unsqueeze(dim=0).repeat(mask.shape[0], 1)

            # print(input_ids.shape)
            # print(mask.shape)
            # print(image_tensor.shape)
            # print(position_ids.shape)

            with torch.no_grad():
                t_output = t_model(
                    input_ids = input_ids.to(device), 
                    attention_mask = mask.to(device),
                    position_ids = position_ids.to(device),
                    use_cache = True,
                    output_attentions = False,
                    output_hidden_states = False,
                    images = image_tensor.to(device),
                    return_dict = True,
                )
                t_result = torch.gather(t_output.logits, 1, mask_indices.view(-1, 1).expand(t_output.logits.shape[0], t_output.logits.shape[2]).unsqueeze(1)).squeeze(1)


            output = model(
                    input_ids = input_ids.to(device), 
                    attention_mask = mask.to(device),
                    position_ids = position_ids.to(device),
                    use_cache = True,
                    output_attentions = False,
                    output_hidden_states = False,
                    images = image_tensor.to(device),
                    return_dict = True,
                )
            result = torch.gather(output.logits, 1, mask_indices.view(-1, 1).expand(output.logits.shape[0], output.logits.shape[2]).unsqueeze(1)).squeeze(1)

            print(result.shape)

            if result.shape != t_result.shape:
                print(result.shape)
                print(t_result.shape)
                continue
            
            loss = loss_func(result,t_result)
            loss.backward()

            with open("../save/loss.txt", "a") as f:
                f.write(str(epoch_min)+" "+str(float(loss))+"\n")

        op.step()

        if epoch_min % 1000 == 0:
            torch.save(model.state_dict(), "../save/"  + "model.bin")

        
   
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