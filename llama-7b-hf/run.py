import argparse
import gdown
from finetune import train
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="output.jsonl")
parser.add_argument('--wandb_project', type=str, default="huggyllama/llama-7b")
parser.add_argument('--wandb_log_model', type=str, default="true")
parser.add_argument('--wandb_run_name', type=str, default="finetune_3_epoch")
args = parser.parse_args()
print(args.data_path)

#load_dataset
url = 'https://drive.google.com/uc?export=download&id=1jbbUtwgwoSQgGnXxzTh-nMReVzEU7ZTU&confirm=t&uuid=d79e2e78-51de-466f-9ceb-3944606141a2&at=AKKF8vwcgi95TGSnSQUNCKx4NTqS:1682865249145'
gdown.download(url,output='output.jsonl', quiet=False)

kwargs = {
    "data_path":args.data_path,
    "wandb_project":args.wandb_project,
    "wandb_log_model":args.wandb_log_model,
    "wandb_run_name":args.wandb_run_name
}

data = train(**kwargs)
