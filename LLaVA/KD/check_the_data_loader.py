import argparse
import sys

import torch
from tqdm import tqdm
sys.path.append("../LLaVA")
sys.path.append("../LLaVA/llava")
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.train.train import LazySupervisedDataset, make_supervised_data_module
from torch.utils.data import DataLoader

def main(args):
    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
    #                                                                        args.load_8bit, args.load_4bit,
    #                                                                        device=args.device)
    

    # print(model.lm_head.parameters())
    # print(model.model.layers)
    # print(model.model.layers.parameters())
    # vision_tower = model.get_vision_tower()
    # args.image_processor = vision_tower.image_processor
    # args.is_multimodal = True
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=args.data_path,
                                data_args=args)

    
    # for i in tqdm(train_dataset):
    #     print(i) 
    #     print(tokenizer.decode(torch.cat([i["input_ids"][:35],i["input_ids"][37:]])))   # 全文
    #     print(i["labels"])   # 需要回答的，其余部分 -100
    #     print(i["images"])   # 图片
    #     break
        
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="../ft/")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="../")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", type=bool, default=True)
    parser.add_argument("--debug", action="store_true")
    
    parser.add_argument("--data-path", type=str, default="../llava_v1_5_mix665k.json")
    parser.add_argument("--image-folder", type=str, default="../")
    parser.add_argument("--mm_use_im_start_end", type=bool, default=False)
    parser.add_argument("--mm_use_im_patch_token", type=bool, default=False)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    args = parser.parse_args()
    main(args)