# compress_VLMs


## BACKGROUND

This is my undergraduate graduation project.

I spent my final year of my undergraduate studies at NUS (National University of Singapore) and completed my graduation project under the guidance of [Wang Xinchao](https://cde.nus.edu.sg/ece/staff/wang-xinchao/). The topic is "Compress the visual language models". 



## SETUP

[Gongfan Fang](https://fangggf.github.io/), he is the direct supervisor of my graduation project, I am very grateful to him.

1. Inspired by [LLaVA](https://github.com/haotian-liu/LLaVA), We use [Sheared-LLaMA](https://github.com/princeton-nlp/LLM-Shearing) instead of the original language model to make a "**TinyLLaVA**".
1. I finish the pretrain and finetune code in LLaVA file.
1. I start to write some code about Knowledge Distillation (in file LLaVA/KD), use 13B LLaVA as teacher, but it is too slow to Knowledge Distillation because I haven't mastered how to write a distributed parallel version yet and my GPU graphics memory is too small.
1. I focus on the [transformers](https://huggingface.co/docs/transformers/index) library, I think the trainer can help me to write the parallel code which can speed up the training by [deepspeed](https://github.com/microsoft/DeepSpeed).
