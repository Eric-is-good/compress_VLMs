import os

if __name__ == "__main__":
    os.system("bash ./start.sh")
    os.system("wandb login 544ae8b5271778aa401fbd7120bd4dba4ca9f808")
    os.system("bash scripts/v1_5/finetune_eric.sh")

    # python3 -m llava.serve.cli \
    # --model-path /opt/data/private/eric/save_model/llava/finetune \
    # --image-file /opt/data/private/eric/save_model/llava/checkpoint/finetune/1.jpg \
    # --load-4bit
    
    # python3 -m llava.serve.controller --host 0.0.0.0 --port 10000
    # python3 -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
    # python3 -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path /opt/data/private/eric/save_model/llava/finetune