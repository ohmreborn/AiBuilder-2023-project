import os
import sys
import requests
import multiprocessing
from typing import Union, List

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from datasets import load_dataset, disable_progress_bar
import torch

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig,
    prepare_model_for_int8_training,

)


class Prompter(object):

    def __init__(self):

        url = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/templates/alpaca.json"
        response = requests.request("GET", url)
        self.template = response.json()

    def generate_prompt(  # แปลง text ให้อยู่ในรูป ที่ ai ต้องการ
        self,
        instruction: str = "Please create an inference question in the style of TOEFL reading comprehension section. Also provide an answer in the format",
        inputs: Union[None, str] = None,  # context
        label: Union[None, str] = None,  # question
    ) -> str:
        if inputs:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=inputs
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        return res


def train(
    # model/data params
    base_model: str = "decapoda-research/llama-7b-hf",  # ชื่อ model ใน hugging Face
    data_path: str = "",  # dataset_path
    output_dir: str = "./lora-alpaca",  # save model ที่ไหน
    # training hyperparams
    batch_size: int = 128,  # จำปรับค่า weight ทุกๆ batch_size step
    micro_batch_size: int = 1,  # batch_size
    num_epochs: int = 3,  # จำนวน epoch
    learning_rate: float = 2e-5,  # learning-rate
    cutoff_len: int = 512,  # จำนวนคำที่ยาวที่สุดในประโยคนั้นมั้ง ไม่แน่ใจ_
    val_set_size: int = 0,  # 2000 # จำนวน ข้อมูลใน validdationset
    # lora hyperparams
    lora_r: int = 8,  # ไม่รู้__
    lora_alpha: int = 16,  # ไม่รู้__ // Alpha => learning rate
    lora_dropout: float = 0.05,  # ไม่รู้__ => check this out please https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9
    lora_target_modules: List[str] = [
        "q_proj",  # fintune เฉพาะ layer นี้
        "v_proj",  # fintune เฉพาะ layer นี้
    ],
    # llm hyperparams
    # faster, but produces an odd training loss curve # ไม่รู้__
    group_by_length: bool = False,
    # wandb params
    # ไม่รู้__   ///////////////// WanDB is just a training monitoring tools. Ignoring is fine :3.
    wandb_project: str = "",
    wandb_run_name: str = "",  # ไม่รู้__
    wandb_log_model: str = "",  # options: false | true # ไม่รู้__
    # either training checkpoint or final adapter # ไม่รู้__ ///// Train จากเช็คพ้อยต่อมั้ย
    resume_from_checkpoint: str = None,
):
    print(
        f"Training Alpaca-LoRA model with params:\n",
        f"base_model: {base_model}\n",
        f"data_path: {data_path}\n",
        f"output_dir: {output_dir}\n",
        f"batch_size: {batch_size}\n",
        f"micro_batch_size: {micro_batch_size}\n",
        f"num_epochs: {num_epochs}\n",
        f"learning_rate: {learning_rate}\n",
        f"cutoff_len: {cutoff_len}\n",
        f"val_set_size: {val_set_size}\n",
        f"lora_r: {lora_r}\n",
        f"lora_alpha: {lora_alpha}\n",
        f"lora_dropout: {lora_dropout}\n",
        f"lora_target_modules: {lora_target_modules}\n",
        f"group_by_length: {group_by_length}\n",
        f"wandb_project: {wandb_project}\n",
        f"wandb_run_name: {wandb_run_name}\n",
        f"wandb_log_model: {wandb_log_model}\n",
        f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
    )
    assert (
        base_model  # เช็คว่า ได้ใส่ ช์่อของ model มาหรือเปล่า
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    assert data_path, "please enter data_path"

    # ปรับค่า weight เมื่อครบ gradient_accumulation_steps iteration ครั้ง
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "auto"
    max_memory = {i: f"{int(mem/1024**3)}GB"for i,
                  mem in enumerate(torch.cuda.mem_get_info())}
    cpu_cores = multiprocessing.cpu_count()

# _________________________________________________________________________________wandb
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
# _________________________________________________________________________________

  # load model
    def load_model(base_model=base_model):

        # load tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        tokenizer.pad_token_id = (0)  # unk. ทำให้ มี token padding คือ UNK
        tokenizer.padding_side = "left"  # Allow batched inference
        # load_model
        quantization_config = BitsAndBytesConfig(
            llm_int8_enable_fp32_cpu_offload=True)
        model = LlamaForCausalLM.from_pretrained(
            base_model,  # decapoda-research/llama-7b-hf
            load_in_8bit=True,  # ไม่รู้_
            torch_dtype=torch.float16,  # ไม่รู้_ /////// Check this out https://moocaholic.medium.com/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407 https://www.quora.com/What-is-the-difference-between-FP16-and-FP32-when-doing-deep-learning
            device_map=device_map,  # ไม่รู้_
            max_memory=max_memory,
            quantization_config=quantization_config,
        )

        # needed for single world model parallel
        model.is_parallelizable = True
        model.model_parallel = True

        # set ค่า requires_grad=False หรือ จะไม่ปรับค่า weight ตอน train
        model = prepare_model_for_int8_training(model)
        config = LoraConfig(
            r=lora_r,  # 8
            lora_alpha=lora_alpha,  # 16
            target_modules=lora_target_modules,  # ["q_proj","v_proj"]
            lora_dropout=lora_dropout,  # 0.5
            bias="none",
            task_type="CAUSAL_LM",
        )
        # ปรับ ค่า weight ในบาง layer ["q_proj","v_proj"]  ให้มีค่า requires_grad=True
        model = get_peft_model(model, config)
        return model, tokenizer

    model, tokenizer = load_model(base_model=base_model)
    prompter = Prompter()

    def tokenize(prompt):  # tokenize
        result = tokenizer(
            prompt['prompt'],
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def format_prompt(data_point):  # create_prompt
        full_prompt = prompter.generate_prompt(
            inputs=data_point['Background:'],
            label=data_point['<human>:_<bot>:']
        )
        return {'prompt': full_prompt, 'length': len(tokenizer.tokenize(full_prompt))}

    # load data
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    # แบ่ง train-test แล้วเอา ประโยคที่มีจำนวนคำ ที่ cutoff_len -1 เพราะ ต้องมี <\s> อยู่หน้าประโยค
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(format_prompt)
        train_data = train_data.filter(
            lambda x: x['length'] <= cutoff_len-1, num_proc=cpu_cores)
        train_data = train_data.map(
            tokenize, batched=False, num_proc=cpu_cores, remove_columns=train_data.column_names)

        val_data = train_val["test"].shuffle().map(format_prompt)
        val_data = val_data.filter(
            lambda x: x['length'] <= cutoff_len-1, num_proc=cpu_cores)
        val_data = val_data.map(
            tokenize, batched=False, num_proc=cpu_cores, remove_columns=train_data.column_names)
    else:

        train_data = data["train"].shuffle().map(
            format_prompt, num_proc=cpu_cores)  # genprompt ให้อยู่ในรูปแบบที่ Ai ต้องการ
        train_data = train_data.filter(
            lambda x: x['length'] <= cutoff_len-1, num_proc=cpu_cores)  # เอาdata ทีมีจำนวนคำเกิน ออก
        train_data = train_data.map(
            tokenize, batched=False, num_proc=cpu_cores, remove_columns=train_data.column_names)
        val_data = None

    # เช็คว่า มีค่า weight กี่ตัวที่ สามารถปรับค่า weight ตอนเทรนมั้ง
    model.print_trainable_parameters()

    train_args = TrainingArguments(  # สร้าง class train-args
        per_device_train_batch_size=micro_batch_size,  # btch_size
        # https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation เหมือนจะ ค่อยๆคำนวนค่า gradient ตามค่าที่ใส่เข้าไปรอบ แล้วค่อยปรับ weight ทีเดียว ไม่รู้_
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,  # learning-rate จะเพิ่มจนถึง ค่า default ของ learning-rate โดย optim.step ทั้งหมด warmup_step ครั้ง แรก จึงค่อยลด
        num_train_epochs=num_epochs,  # จำนวน epoch
        learning_rate=learning_rate,  # ค่า learning-rate
        fp16=True,
        logging_steps=10,  # ไม่รู้_ //////////// แสดงผลตอนเทรนทุกๆ 10 step gradient descent
        optim="adamw_torch",  # ชื่อ optimizer มั้ง_ /////// yes!!
        # ไม่รู้_ ///////// Evaluate model if it has validation set.
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        # ไม่รู้_ //////////////////////////////// save model based on epoch? steps?
        save_strategy="steps",
        # ไม่รู้_ ///////// Evaluate model if it has validation set.
        eval_steps=200 if val_set_size > 0 else None,
        save_steps=200,  # ไม่รู้_ ///////////////////// Save model every 200 optimizer.step()
        output_dir=output_dir,  # ไม่รู้_ ////////////////////////// Where to save model
        save_total_limit=3,
        # ไม่รู้_   ///////// Evaluate model if it has validation set.
        load_best_model_at_end=True if val_set_size > 0 else False,
        group_by_length=group_by_length,  # ไม่รู้ _
        report_to="wandb",  # ไม่รู้_
        run_name=wandb_run_name,  # ไม่รู้_
    )

    trainer = Trainer(
        model=model,  # model ที่จะเอาไปเทรน
        train_dataset=train_data,  # data ใน train-set
        eval_dataset=val_data,  # data ใน validation-set
        args=train_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True  # ไม่รู้_
        ),
    )

    # ไม่รู้__ //// Use GPU cache. Ignore is fine.
    model.config.use_cache = False

# train-ai ปกติ
    trainer.train(
        resume_from_checkpoint=resume_from_checkpoint  # save checkpoint
    )

    model.save_pretrained(output_dir)  # save pretrained มั้ง_ /// yes

    print("training sucess")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str,
                        default="decapoda-research/llama-7b-hf")
    parser.add_argument('--data_path', type=str, default="output.jsonl")
    parser.add_argument('--output_dir', type=str, default="./lora-alpaca")

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--micro_batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--cutoff_len', type=int, default=1024)
    parser.add_argument('--val_set_size', type=int, default=0)

    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('-l', '--lora_target_modules',
                        default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument('--wandb_project', type=str,
                        default="huggyllama-llama-7b")
    parser.add_argument('--wandb_log_model', type=str, default="true")
    parser.add_argument('--wandb_run_name', type=str,
                        default="finetune_llama")

    parser.add_argument('--group_by_length', type=bool, default=False)
    args = parser.parse_args()

    assert args.data_path.endswith(
        '.jsonl'), "please enter path with .jsonl like output.jsonl"

    # create training argument on dictionary format
    kwargs = vars(args)
    disable_progress_bar()
    train(**kwargs)
