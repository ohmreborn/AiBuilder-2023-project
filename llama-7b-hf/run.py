import argparse
import gdown
from finetune import train
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
args = parser.parse_args()
print(args.data_path)

#load_dataset
url = 'https://drive.google.com/uc?export=download&id=1jbbUtwgwoSQgGnXxzTh-nMReVzEU7ZTU&confirm=t&uuid=d79e2e78-51de-466f-9ceb-3944606141a2&at=AKKF8vwcgi95TGSnSQUNCKx4NTqS:1682865249145'
output = 'output.jsonl'
gdown.download(url, output, quiet=False)

kwargs = {
    "data_path":args.data_path,
    "wandb_project":"huggyllama/llama-7b",
    "wandb_log_model":"true",
    "wandb_run_name":"finetune_3_epoch"
}

data = train(**kwargs)
