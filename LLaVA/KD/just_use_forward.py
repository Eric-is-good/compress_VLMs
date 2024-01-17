import argparse
import torch
import sys
sys.path.append("../LLaVA/llava")
sys.path.append("../LLaVA")
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.serve.cli import load_image
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                           args.load_8bit, args.load_4bit,
                                                                           device=args.device)
    print("##########################################")
    print(tokenizer)
    print("##########################################")
    print(model)
    print("##########################################")
    print(image_processor)
    print("##########################################")
    print(context_len)

    # if 'llama-2' in model_name.lower():
    #     conv_mode = "llava_llama_2"
    # elif "v1" in model_name.lower():
    #     conv_mode = "llava_v1"
    # elif "mpt" in model_name.lower():
    #     conv_mode = "mpt"
    # else:
    #     conv_mode = "llava_v0"
    args.conv_mode = "v1"

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(args.image_file)
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)  # [1,3,336,336]
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        print(input_ids.dtype) # torch.int64
        print(tokenizer.decode(torch.cat([input_ids[0,:35],input_ids[0,37:]])))
        print(image_tensor.dtype) # torch.float16

        output = model(
            input_ids = input_ids, 
            attention_mask = torch.ones_like(input_ids).to(model.device),
            position_ids = torch.arange(0,input_ids.shape[1]).unsqueeze(dim=0).to(model.device),
            use_cache = True,
            output_attentions = False,
            output_hidden_states = False,
            images = image_tensor,
            return_dict = True,
        )
        
        print(type(output)) # keys ['logits', 'past_key_values']
        print(output.logits.shape) # [1, n ,32000]
        logits = torch.softmax(output.logits[0][-1],-1)
        
        # logits = torch.softmax(output.logits[0],-1)
        # print(logits)
        # id = torch.argmax(logits, dim=-1)
        id = torch.argmax(logits)
        # text = tokenizer.decode(id)
        # print(text)
        
        # loss = torch.sum(logits)
        # logits.retain_grad()
        # loss.backward()
        # print(logits.grad)
        
        
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="../ft")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="../test.jpg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", type=bool, default=False)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
