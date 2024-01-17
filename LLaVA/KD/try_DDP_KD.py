import argparse
import sys
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel 

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

# # 设置环境变量
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5679'
# dist.init_process_group(backend='gloo', init_method='env://', rank=0, world_size=int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1)

def main(args):
    # 新增：从外面得到local_rank参数
    local_rank = int(args.local_rank)

    # 新增：DDP backend初始化
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
    device = torch.device("cuda", local_rank)

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,args.load_8bit, args.load_4bit, device=args.device)
    model.train()

    t_model_name = get_model_name_from_path(args.t_model_path)
    t_tokenizer, t_model, t_image_processor, t_context_len = load_pretrained_model(args.t_model_path, args.model_base, t_model_name,args.load_8bit, args.load_4bit, device=args.device)
    t_model.eval()

    vision_tower = model.get_vision_tower()
    args.image_processor = vision_tower.image_processor
    args.is_multimodal = True

    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,data_path=args.data_path, data_args=args)

    # parameters
    # opt_model = model.model
    # decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    # decay_parameters = [name for name in decay_parameters if "bias" not in name]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [
    #             p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
    #         ],
    #         "weight_decay": 0.0,
    #     },
    #     {
    #         "params": [
    #              p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
    #         ],
    #         "weight_decay": 0.0,
    #     },
    #     {
    #         "params": [
    #             p for n, p in model.lm_head.named_parameters() if p.requires_grad
    #         ],
    #         "weight_decay": 0.0,
    #     }
    # ]

    model = DistributedDataParallel(model)
    t_model = DistributedDataParallel(t_model)
    
    head = list(model.lm_head.parameters())
    llm = list(model.model.layers.parameters())

    # op = torch.optim.SGD(params=[{"params":head}, {"params":llm}], lr=1e-4)
    op = torch.optim.SGD(params=[{"params":head}], lr=1e-4)
    # op = torch.optim.AdamW(model.lm_head.parameters(), lr=1e-4, weight_decay=0)
    loss_func = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    batch_size = 4
    epoch_min = 0
    step_mun = 0

    op.zero_grad()
    for i in tqdm(train_dataset):
        epoch_min = epoch_min + 1
        labels = i["labels"] 
        full_input_ids = i["input_ids"].unsqueeze(dim=0)
        image_tensor_ = i["images"].half().unsqueeze(dim=0)
        has_img = i["has_img"]

        # 找到所有大于-100的索引
        big_mask_indices = (labels > -100).nonzero(as_tuple=False).squeeze()[:2000]
        # big_mask_indices = big_mask_indices[torch.randperm(big_mask_indices.shape[0])]
        max_index = torch.max(big_mask_indices)
        big_mask = (torch.arange(labels.shape[0]).unsqueeze(0) <= (big_mask_indices-1).unsqueeze(1))

        in_batch_time = (len(big_mask_indices) + batch_size - 1) // batch_size

        # print(labels)
        # print(full_input_ids)
        # print(big_mask_indices)

        for i in range(in_batch_time):
            step_mun = step_mun + 1
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
                index = (mask_indices+574).unsqueeze(1).repeat(1,32000).reshape(bs,1,32000)
            else:
                index = (mask_indices-1).unsqueeze(1).repeat(1,32000).reshape(bs,1,32000)

            # t_result = F.softmax(torch.gather(t_output.logits, 1, index).squeeze(1), dim=1)
            t_result = torch.gather(t_output.logits, 1, index).squeeze(1)

            # print(t_result.shape)
            # _, pred = torch.max(t_result, 1)
            # print(pred) 
            # break

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
            # result = F.softmax(torch.gather(output.logits, 1, index).squeeze(1), dim=1) 
            result = torch.gather(output.logits, 1, index).squeeze(1)

            loss = loss_func(result, t_result)
            loss.backward()

            with open("../save/loss.txt", "a") as f:
                f.write(str(epoch_min)+" "+str(float(loss.item()))+"\n")
        
        # break
            if step_mun == 10:
                step_mun = 0
                op.step()
                op.zero_grad()
        # torch.cuda.empty_cache()

        if epoch_min % 100 == 0:
            torch.save(model.state_dict(), "../save/"  + "model.bin")
    
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

    parser.add_argument("--local-rank", default=-1)

    args = parser.parse_args()
    main(args)