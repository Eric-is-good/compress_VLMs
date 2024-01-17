import argparse
import sys
import os
import torch
from tqdm import tqdm
sys.path.append("../LLaVA")
sys.path.append("../LLaVA/llava")
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.train.train import LazySupervisedDataset, make_supervised_data_module, ModelArguments, DataArguments, TrainingArguments
from torch.utils.data import DataLoader
from transformers.trainer import get_parameter_names, ALL_LAYERNORM_LAYERS
import torch.nn as nn
import traceback
import torch.nn.functional as F
import deepspeed
from transformers import Trainer
import transformers

import os
os.environ["WANDB_DISABLED"]="true"

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

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    model_name = get_model_name_from_path("../ft/")
    tokenizer, model, image_processor, context_len = load_pretrained_model("../ft/", None, model_name,False, False, device="cuda")
    model.train()

    t_model_name = get_model_name_from_path("../13B/")
    t_tokenizer, t_model, t_image_processor, t_context_len = load_pretrained_model("../13B/", None, t_model_name,False, False, device="cuda")
    t_model.eval()

    vision_tower = model.get_vision_tower()
    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,data_path="../llava_v1_5_mix665k.json", data_args=data_args)

    loss_func = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    batch_size = 4

    class DistillationTrainer(Trainer):
        def create_optimizer(self):
            head = list(self.model.lm_head.parameters())
            llm = list(self.model.model.layers.parameters())
            optimizer_grouped_parameters = [{"params":head}]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
            )
            return self.optimizer

        def compute_loss(self, model, i):
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
                
                loss += loss_func(result, t_result)

            return loss

    trainer = DistillationTrainer(
        model = model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()
    trainer.save_state()

if __name__ == "__main__":
    train()
    
    