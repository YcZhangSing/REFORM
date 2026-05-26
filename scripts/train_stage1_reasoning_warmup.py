
import os
import math
import time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
import json
import argparse
from datetime import datetime
import friendlywords as fw
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)
import torch.nn.functional as F
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch
import wandb

from reform.rom_dataset import ROMDatasetForTraining, load_rom_json
from sklearn.feature_extraction.text import TfidfVectorizer
from reform import train_utils
from inspect import isfunction
globals().update({
    name: func for name, func in vars(train_utils).items() if isfunction(func)
})
import torch.multiprocessing as mp
print("[INFO] multiprocessing start method:", mp.get_start_method())



train_data = []
val_data = []
train_epoch  = 4
train_bs = 4
need_replace_lmHead = True
accumulation_steps = 2
save_epoch_step = 1
need_wandb = False
train_js = None
val_js_list = []
dataset_root = None
dataloader_num_workers = 0

main_lr = 1e-5
florence_init_pth = None
reform_model_path = os.path.join(PROJECT_ROOT, "models")
output_root = os.path.join(PROJECT_ROOT, "outputs", "stage1_reasoning_warmup")
logged_task_name = "REFORM_stage1_reasoning_warmup"


train_time = datetime.now().strftime("%Y%m%d_%H%M%S")
local_train_name = f'{logged_task_name}_{train_time}'

def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def load_runtime_config_from_env():
    global train_js, val_js_list, dataset_root, florence_init_pth, reform_model_path
    global output_root, need_replace_lmHead, need_wandb, logged_task_name, dataloader_num_workers

    train_js = os.environ.get("REFORM_TRAIN_JSON", train_js)
    val_jsons = os.environ.get("REFORM_VAL_JSONS")
    if val_jsons:
        val_js_list = val_jsons.split(os.pathsep)
    dataset_root = os.environ.get("REFORM_DATASET_ROOT", dataset_root)
    florence_init_pth = os.environ.get("REFORM_MODEL_PATH", florence_init_pth)
    reform_model_path = os.environ.get("REFORM_REFORM_MODEL_PATH", reform_model_path)
    output_root = os.environ.get("REFORM_OUTPUT_ROOT", output_root)
    logged_task_name = os.environ.get("REFORM_TASK_NAME", logged_task_name)
    need_replace_lmHead = _env_flag("REFORM_REPLACE_LM_HEAD", need_replace_lmHead)
    need_wandb = _env_flag("REFORM_WANDB", need_wandb)
    dataloader_num_workers = int(os.environ.get("REFORM_NUM_WORKERS", dataloader_num_workers))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def get_loss_weights(current_step, total_steps=30000, reason_max=1.0, reason_min=0.1):
    if current_step>total_steps:
        reason_weight = 0.1
        answer_weight = 1
    else:
        progress = current_step / total_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        reason_weight = reason_min + (reason_max - reason_min) * cosine_decay
        answer_weight = 1.1 - reason_weight
    return reason_weight, answer_weight


def train_model(rank, world_size, dataset_name, batch_size=6, use_lora=False, epochs=10, lr=1e-6, eval_steps=10, run_name=None, max_val_item_count=1000):
    load_runtime_config_from_env()
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    print(f'使用这些  {device}  设备开始训练')
    

    criterion = torch.nn.CrossEntropyLoss() 
    
    if run_name is None:
        run_name = fw.generate(2, separator="_")

    option_labels = [
            torch.tensor([0, 0, 0, 0, 0]).to(device),
            torch.tensor([1, -0.25, -0.25, -0.25, -0.25]).to(device),
            torch.tensor([-0.25, 1, -0.25, -0.25, -0.25]).to(device),
            torch.tensor([-0.25, -0.25, 1, -0.25, -0.25]).to(device),
            torch.tensor([-0.25, -0.25, -0.25, 1, -0.25]).to(device),
            
            torch.tensor([1, -0.66, -0.66, -0.66, 1]).to(device),
            torch.tensor([-0.66, 1, -0.66, -0.66, 1]).to(device),
            torch.tensor([-0.66, -0.66, 1, -0.66, 1]).to(device),
            torch.tensor([-0.66, -0.66, -0.66, 1, 1]).to(device),
            torch.tensor([-0.25, -0.25, -0.25, -0.25, 1]).to(device),
            
    ]
    
    options = [
            "A. No.", # #[0,0,0,0,0]
            "B. Image: Face swap; Text: No.",# #[1,0,0,0,0]
            "C. Image: Face attribute; Text: No.",# #[0,1,0,0,0]
            "D. Image: Whole generated; Text: No.",# #[0,0,1,0,0]
            "E. Image: Inpainted background; Text: No.",# #[0,0,0,1,0]
            "F. Image: Face swap; Text: Fully rewritten.",# #[1,0,0,0,1]
            "G. Image: Face attribute; Text: Fully rewritten.",# #[0,1,0,0,1]
            "H. Image: Whole generated; Text: Fully rewritten.",# #[0,0,1,0,1]
            "I. Image: Inpainted background; Text: Fully rewritten.",# #[0,0,0,1,1]
            "J. Image: No; Text: Fully rewritten.",# #[0,0,0,0,1]
    ]
    
    vectorizer = TfidfVectorizer().fit(options)
    option_vectors = vectorizer.transform(options).toarray()
    
    if need_replace_lmHead:

        if reform_model_path is None:
            raise ValueError("--reform-model-path is required when --replace-lm-head is enabled")
        reform_model = AutoModelForCausalLM.from_pretrained(reform_model_path, trust_remote_code=True)

        reform_model_lm_head_weight = reform_model.language_model.lm_head.weight.data.clone()
        reform_model_logits_bias = reform_model.language_model.final_logits_bias.data.clone() 



        del reform_model
        torch.cuda.empty_cache()
        
        # Load the model and processor
        model = AutoModelForCausalLM.from_pretrained(
            florence_init_pth, trust_remote_code=True
        ).to(device)
        

        assert reform_model_lm_head_weight.shape == model.language_model.lm_head.weight.shape, \
            f"Shape mismatch: {reform_model_lm_head_weight.shape} vs {model.language_model.lm_head.weight.shape}"
        assert reform_model_logits_bias.shape == model.language_model.final_logits_bias.shape, \
            f"Bias shape mismatch: {reform_model_logits_bias.shape} vs {model.language_model.final_logits_bias.shape}"


        with torch.no_grad():
            model.language_model.lm_head.weight.copy_(reform_model_lm_head_weight)
            model.language_model.final_logits_bias.copy_(reform_model_logits_bias)

        print("REFORM model lm_head and final_logits_bias restored successfully")

    else:
        # Load the model and processor
        model = AutoModelForCausalLM.from_pretrained(
            florence_init_pth, trust_remote_code=True
        ).to(device)
        
    

    
    def Unfreeze_model(model):

        for param in model.language_model.model.encoder.parameters():
            param.requires_grad = True

        for param in model.language_model.lm_head.parameters():
            param.requires_grad = True


        for param in model.language_model.model.decoder.parameters():
            param.requires_grad = True


        model.language_model.model.learnable_tokens.requires_grad = True



        for param in model.language_model.model.classifier.parameters():
            param.requires_grad = True


        model.language_model.model.Dubl_alpha.requires_grad = True


        for param in model.language_model.model.attn_linear.parameters():
            param.requires_grad = True
            
        for param in model.language_model.model.attn_weight.parameters():
            param.requires_grad = True


        for param in model.language_model.model.Bbox_Verification.parameters():
            param.requires_grad = True


        for param in model.language_model.model.gnn.parameters():
            param.requires_grad = True
        for param in model.language_model.model.gnn2.parameters():
            param.requires_grad = True


    Unfreeze_model(model)
    

    torch.cuda.synchronize()
    
    processor = AutoProcessor.from_pretrained(florence_init_pth, trust_remote_code=True)
    not_init_wandb = True   
    model = DDP(model, device_ids=[rank])

    if dataset_name == 'ROM':
        train_data = load_rom_json(train_js, dataset_root)
        for val_js in val_js_list:
            val_data.extend(load_rom_json(val_js, dataset_root))
            
        train_dataset = ROMDatasetForTraining(split='train',data=train_data)
        val_datasets = {"ROM": ROMDatasetForTraining(split='validation',data=val_data)}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    
    # Create DataLoaders
    train_loader, val_loaders = create_data_loaders(
        train_dataset,
        val_datasets,
        batch_size,
        dataloader_num_workers,
        rank,
        world_size,
        processor,
        device,
    )

    optimizer = AdamW(model.parameters(), lr=lr)
        
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=5000,
        num_training_steps=num_training_steps,
    )
    
    global_step = 0
    beast_eva_ACC = 0    

    for epoch in range(epochs):
        # Training phase
        model.train()        

        loss_list = []
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch + 1}/{epochs}")
        for step, batch in progress_bar:
            inputs, answers,fake_image_box,reason  = batch

            # Prepare the input and target tensors
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            

            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                max_length=800,
            ).input_ids.to(device)
            

            reason_labels = processor.tokenizer(
                text=reason,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                max_length=800,                
            ).input_ids.to(device)

            
            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels, reason_labels = reason_labels
            )
            

            reason_weight, answer_weight = get_loss_weights(global_step)

            # # ###
            # print(f'outputs.loss is {outputs.loss}')
            # print(f'outputs.reason_loss is {outputs.reason_loss}')
            # # break

            total_loss = answer_weight*outputs.loss + reason_weight*outputs.reason_loss
            
            if torch.isnan(total_loss):
                raise ValueError("reason_loss became NaN, terminating training.")
            
            

            Binary_lables = []
            for label in answers:
                if label.lower().startswith('a'):
                    Binary_lables.append(1)
                else:
                    Binary_lables.append(0)

            Binary_lables = torch.tensor(Binary_lables, dtype=torch.long).to(device)

            logits_list = outputs.classification_logits_list

            for i,logits in enumerate(logits_list):
                if logits is not None:
                    torch.cuda.synchronize()
                    if i == 0:
                        temp_loss0 = criterion(logits,Binary_lables)
                        total_loss += 0.1*temp_loss0
                        loss_list.append(temp_loss0)
                        
                    if i == 1:
                        output_coords = logits.to(device)
                        tensor_fake_image_box = torch.cat(fake_image_box, dim=0).reshape(len(fake_image_box), -1).to(device)
                        loss_bbox, loss_giou = get_bbox_loss(output_coords, tensor_fake_image_box)
                        total_loss += 0.1*(loss_bbox+loss_giou)
                        loss_list.append(loss_bbox)
                        loss_list.append(loss_giou)
                else:
                    print(f'Attention!! !!模型返回的logits[{i}]是None!!! ')

            total_loss = total_loss / accumulation_steps            
            # -------------------------------------------



            # -------------------------------------------
            if (step + 1) % accumulation_steps != 0:
                with model.no_sync():
                    total_loss.backward()
            else:
                total_loss.backward()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            if rank == 0 and need_wandb:
                if not_init_wandb:
                    not_init_wandb = False
                    if os.getenv("WANDB_API_KEY"):
                        wandb.login(key=os.environ["WANDB_API_KEY"])
                    wandb.init(project=logged_task_name, name=run_name)
                    wandb.config.update({
                        "dataset": dataset_name,
                        "batch_size": batch_size,
                        "use_lora": use_lora,
                        
                        "epochs": epochs,
                        "learning_rate": lr,
                        "eval_steps": eval_steps,
                        "world_size": world_size,
                    })
                wandb.log({"step": global_step + 1, "step_weighted_sum_loss": total_loss.item()})
                wandb.log({"step": global_step + 1, "step_avg_answer_loss": outputs.loss.item()})
                wandb.log({"step": global_step + 1, "step_avg_reason_loss": outputs.reason_loss.item()})
                
                
                wandb.log({"step": global_step + 1, "step_avg_LearnableToken_loss": loss_list[0].item()})
                wandb.log({"step": global_step + 1, "step_avg_bbox_loss": loss_list[1].item()})
                wandb.log({"step": global_step + 1, "step_avg_giou_loss": loss_list[2].item()})
            
            loss_list.clear()    
            global_step += 1
            if global_step % eval_steps == 0:
                eva_ACC = evaluate_model(rank, world_size, model, val_loaders, device, processor, global_step, max_val_item_count,option_vectors,vectorizer,options,option_labels,need_wandb)
                out_put_prefix = output_root
                
                if beast_eva_ACC < eva_ACC:   
                    beast_eva_ACC = eva_ACC
                    output_dir = os.path.join(out_put_prefix, f"train_{train_time}_{logged_task_name}/best_ckpt")
                    os.makedirs(output_dir, exist_ok=True)
                    save_model(model, processor, output_dir, optimizer=None, lr_scheduler=None, epoch=None)

                    best_info_path = os.path.join(output_dir, "best_info.txt")
                    
                    with open(best_info_path, "w") as f:
                        f.write(f"Best Evaluation Accuracy: {beast_eva_ACC:.4f}\n")
                        f.write(f"Training Step: {global_step}\n")
                        f.write(f"Epoch: {epoch+1}\n")
                        
                            
        # Save model checkpoint
        if rank == 0 and (epoch+1)%save_epoch_step==0:  # Only the main process saves the checkpoint
            
            out_put_prefix = output_root
            output_dir = os.path.join(out_put_prefix, f"train_{train_time}_{logged_task_name}/epoch_{epoch+1}")
            os.makedirs(output_dir, exist_ok=True)
            
            save_model(model, processor, output_dir, optimizer=optimizer, lr_scheduler=lr_scheduler, epoch=epoch)
            

    # Finish the wandb run
    if rank == 0 and need_wandb:
        wandb.finish()

    cleanup()


def main(time_label):
    global train_js, val_js_list, dataset_root, florence_init_pth, reform_model_path
    global output_root, need_replace_lmHead, need_wandb, logged_task_name, local_train_name
    global dataloader_num_workers

    parser = argparse.ArgumentParser(description="Stage 1: Cognitive Reasoning Warm-up")
    parser.add_argument("--dataset", type=str, default="ROM", choices=["ROM"], help="Dataset to train on")
    parser.add_argument("--train-json", required=True, help="ROM training meta.json")
    parser.add_argument("--val-json", nargs="+", required=True, help="One or more ROM validation meta.json files")
    parser.add_argument("--dataset-root", default=None, help="Root of REFORM_ROMdataset for relative image paths")
    parser.add_argument("--model-path", required=True, help="Initial REFORM checkpoint")
    parser.add_argument("--reform-model-path", default=reform_model_path, help="Modified REFORM model directory used to restore lm_head")
    parser.add_argument("--output-dir", default=output_root, help="Directory for checkpoints and logs")
    parser.add_argument("--task-name", default=logged_task_name)
    parser.add_argument("--batch-size", type=int, default=train_bs, help="Batch size for training")
    parser.add_argument("--use-lora", action='store_true', help="Use LoRA if this flag is passed")
    parser.add_argument("--epochs", type=int, default=train_epoch, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=main_lr, help="Learning rate")
    parser.add_argument("--eval-steps", type=int, default=4000, help="Number of steps between evaluations, 4kstep, when bs is 4, is about 50min")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for wandb")
    parser.add_argument("--max-val-item-count", type=int, default=4900, help="Maximum number of items to evaluate on during validation")
    parser.add_argument("--replace-lm-head", action=argparse.BooleanOptionalAction, default=need_replace_lmHead)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-base-url", default=None)
    parser.add_argument("--num-workers", type=int, default=dataloader_num_workers)
    parser.add_argument("--world-size", type=int, default=None)
    args = parser.parse_args()

    train_js = args.train_json
    val_js_list = args.val_json
    dataset_root = args.dataset_root
    florence_init_pth = args.model_path
    reform_model_path = args.reform_model_path
    output_root = args.output_dir
    need_replace_lmHead = args.replace_lm_head
    need_wandb = args.wandb
    logged_task_name = args.task_name
    local_train_name = args.run_name or f"{logged_task_name}_{time_label}"
    dataloader_num_workers = args.num_workers
    if args.wandb_base_url:
        os.environ["WANDB_BASE_URL"] = args.wandb_base_url
    os.environ["REFORM_TRAIN_JSON"] = train_js
    os.environ["REFORM_VAL_JSONS"] = os.pathsep.join(val_js_list)
    if dataset_root:
        os.environ["REFORM_DATASET_ROOT"] = dataset_root
    os.environ["REFORM_MODEL_PATH"] = florence_init_pth
    if reform_model_path:
        os.environ["REFORM_REFORM_MODEL_PATH"] = reform_model_path
    os.environ["REFORM_OUTPUT_ROOT"] = output_root
    os.environ["REFORM_TASK_NAME"] = logged_task_name
    os.environ["REFORM_REPLACE_LM_HEAD"] = str(need_replace_lmHead)
    os.environ["REFORM_WANDB"] = str(need_wandb)
    os.environ["REFORM_NUM_WORKERS"] = str(dataloader_num_workers)
    

    world_size = args.world_size or torch.cuda.device_count()
    # world_size = 1
    
    mp.spawn(
        train_model,
        args=(world_size, args.dataset, args.batch_size, args.use_lora, args.epochs, args.lr, args.eval_steps, local_train_name, args.max_val_item_count),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    
    main(train_time)
