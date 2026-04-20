'''
由于从 Pretrained reason_decoder REFORM 加载权重
所以decoder_2相关的参数不必再显式从decoder复制
注意观察训练时的前排输出，是否有未初始化的权值报错
'''

import os
import math
import time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
## 使用wandb来管理模型训练日志
from reform.rom_dataset import ROMDatasetForTraining, load_rom_json
from sklearn.feature_extraction.text import TfidfVectorizer
from reform import train_utils
from inspect import isfunction
globals().update({
    name: func for name, func in vars(train_utils).items() if isfunction(func) ## 动态导入train_utils中的函数
})
import torch.multiprocessing as mp
print("[INFO] multiprocessing start method:", mp.get_start_method())

###==<--------------参数配置区(start)--------------->==
### bs 太小就会nan
train_data = []
val_data = []
train_epoch  = 13
train_bs = 4
need_replace_lmHead = True
accumulation_steps = 2 ## 注意，accumulation_steps虽好，但是会在一定程度上增加显存占用，目前accumulation_steps=8会在训练中爆显存
save_epoch_step = 1
need_wandb = False
consist_warm_step = 3e4   ## 在哪一步加入一致性损失
reason_answer_warmup=3e4  ## Reason-Answer 权重转换的步数
## <----------- margin需要在 在 florence_init_pth 里设置好 ----------- >
margin = 0 ## 在模型代码里改！
## <----------- margin需要在 在 florence_init_pth 里设置好 ----------- >
num_workers = 8

train_js = None
val_js_list = []
dataset_root = None

main_lr = 1e-5
florence_init_pth = None
florence_base_path = None
output_root = os.path.join(PROJECT_ROOT, "outputs", "stage2_joint_finetune")
consist_warm_step_str = f"{consist_warm_step:.0e}".replace('+0', '') ## 科学计数法转str
reason_answer_warmup_str = f"{reason_answer_warmup:.0e}".replace('+0', '') ## 科学计数法转str

logged_task_name = f'REFORM_stage2_joint_finetune_margin{margin}_consistAdd{consist_warm_step}_RAwarm{reason_answer_warmup_str}'
###==<--------------参数配置区(end)--------------->==


def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def load_runtime_config_from_env():
    global train_js, val_js_list, dataset_root, florence_init_pth, florence_base_path
    global output_root, need_replace_lmHead, need_wandb, logged_task_name, num_workers
    global consist_warm_step, reason_answer_warmup

    train_js = os.environ.get("REFORM_TRAIN_JSON", train_js)
    val_jsons = os.environ.get("REFORM_VAL_JSONS")
    if val_jsons:
        val_js_list = val_jsons.split(os.pathsep)
    dataset_root = os.environ.get("REFORM_DATASET_ROOT", dataset_root)
    florence_init_pth = os.environ.get("REFORM_MODEL_PATH", florence_init_pth)
    florence_base_path = os.environ.get("REFORM_FLORENCE_BASE_PATH", florence_base_path)
    output_root = os.environ.get("REFORM_OUTPUT_ROOT", output_root)
    logged_task_name = os.environ.get("REFORM_TASK_NAME", logged_task_name)
    need_replace_lmHead = _env_flag("REFORM_REPLACE_LM_HEAD", need_replace_lmHead)
    need_wandb = _env_flag("REFORM_WANDB", need_wandb)
    num_workers = int(os.environ.get("REFORM_NUM_WORKERS", num_workers))
    consist_warm_step = float(os.environ.get("REFORM_CONSIST_WARM_STEP", consist_warm_step))
    reason_answer_warmup = float(os.environ.get("REFORM_REASON_ANSWER_WARMUP", reason_answer_warmup))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def get_loss_weights(current_step, total_steps=None, reason_max=1.0, reason_min=0.1):
    """
    在 total_steps =的训练区间内
    reason loss 权重从 1.0 平滑衰减到 0.1
    answer loss 权重则从约 0.1 或 0.2 上升到 1.0
    consist loss 的权重从约1e-5上升到0.1
    形成一个渐进的“主次权重切换”过程。
    """
    if total_steps is None:
        total_steps = reason_answer_warmup
    if current_step>total_steps:
        reason_weight = 0.1
        answer_weight = 1
    else:
        progress = current_step / total_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        reason_weight = reason_min + (reason_max - reason_min) * cosine_decay
        answer_weight = 1.1 - reason_weight
        
    return reason_weight, answer_weight


def train_model(rank, world_size, dataset_name, train_time, batch_size=6, use_lora=False, epochs=10, lr=1e-6, eval_steps=10, run_name=None, max_val_item_count=1000):
    load_runtime_config_from_env()
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    print(f'使用这些  {device}  设备开始训练')
    
    # 实例化用于二分类损失计算的交叉熵损失函数
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
        ## 替换model中的lm_head相关权重
        if florence_base_path is None:
            raise ValueError("--florence-base-path is required when --replace-lm-head is enabled")
        base_model = AutoModelForCausalLM.from_pretrained(florence_base_path, trust_remote_code=True)
        # 1️⃣ 深拷贝 Florence-2 的 lm_head 参数
        base_lm_head_weight = base_model.language_model.lm_head.weight.data.clone()
        base_logits_bias = base_model.language_model.final_logits_bias.data.clone() 


        # 删除 base_model 节省显存
        del base_model
        torch.cuda.empty_cache()
        
        # Load the model and processor
        model = AutoModelForCausalLM.from_pretrained(
            florence_init_pth, trust_remote_code=True
        ).to(device)
        
        # 校验形状
        assert base_lm_head_weight.shape == model.language_model.lm_head.weight.shape, \
            f"Shape mismatch: {base_lm_head_weight.shape} vs {model.language_model.lm_head.weight.shape}"
        assert base_logits_bias.shape == model.language_model.final_logits_bias.shape, \
            f"Bias shape mismatch: {base_logits_bias.shape} vs {model.language_model.final_logits_bias.shape}"

        # 执行覆盖
        with torch.no_grad():
            model.language_model.lm_head.weight.copy_(base_lm_head_weight)
            model.language_model.final_logits_bias.copy_(base_logits_bias)

        print("✅ Florence lm_head 与 final_logits_bias 已成功恢复为原始权重")

    else:
        # Load the model and processor
        model = AutoModelForCausalLM.from_pretrained(
            florence_init_pth, trust_remote_code=True
        ).to(device)
        
    

    
    def Unfreeze_model(model):
        # 解冻 encoder
        for param in model.language_model.model.encoder.parameters():
            param.requires_grad = True
        # answer encoder 的 lm也要解冻
        for param in model.language_model.lm_head.parameters():
            param.requires_grad = True

        # 解冻 decoder
        for param in model.language_model.model.decoder.parameters():
            param.requires_grad = True

        # 解冻 learnable_tokens
        model.language_model.model.learnable_tokens.requires_grad = True

        # 解冻 classifier

        for param in model.language_model.model.classifier.parameters():
            param.requires_grad = True

        # 解冻 Dubl_alpha
        model.language_model.model.Dubl_alpha.requires_grad = True

        # 解冻 attn_linear 和 attn_weight
        for param in model.language_model.model.attn_linear.parameters():
            param.requires_grad = True
            
        for param in model.language_model.model.attn_weight.parameters():
            param.requires_grad = True

        # 解冻 Bbox_Verification
        for param in model.language_model.model.Bbox_Verification.parameters():
            param.requires_grad = True

        # 解冻 gnn 和 gnn2
        for param in model.language_model.model.gnn.parameters():
            param.requires_grad = True
        for param in model.language_model.model.gnn2.parameters():
            param.requires_grad = True

    # 保险起见，显示执行解冻操作
    Unfreeze_model(model)
    

    # # --- [新增] 参数统计 (仅 rank 0 打印) ---
    # if rank == 0:
    #     total_params = sum(p.numel() for p in model.parameters())
    #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    #     print(f"\n{'='*20} Model Stats (After Unfreeze) {'='*20}")
    #     print(f"Total Params:     {total_params:,} ({total_params/1e6:.2f} M)")
    #     print(f"Trainable Params: {trainable_params:,} ({trainable_params/1e6:.2f} M)")
    #     print(f"Trainable Ratio:  {trainable_params/total_params:.2%}")
    #     print(f"{'='*58}\n")
    # # ----------------------------------------


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
        num_workers,
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
        ##loss_list归零初始化
        loss_list = []
        
        # --- [新增] 速度测试初始化变量 ---
        perf_start_time = 0
        perf_sample_count = 0
        # -------------------------------
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch + 1}/{epochs}")
        for step, batch in progress_bar:
            inputs, answers,fake_image_box,reason  = batch
            

            # Prepare the input and target tensors
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            
            ## 主decoder的answer label
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                max_length=800,
            ).input_ids.to(device)
            
            ## decoder_2的reason label
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
            
            ## 慢慢降低reason_weight的权重，同时提升answer_weight的权重
            reason_weight, answer_weight = get_loss_weights(global_step)

            # # ###
            # print(f'outputs.loss is {outputs.loss}')
            # print(f'outputs.reason_loss is {outputs.reason_loss}')
            # # break
            
            total_loss = answer_weight*outputs.loss + reason_weight*outputs.reason_loss
            
            ## 正确一致性损失            
            if global_step > consist_warm_step:
                total_loss += 0.1*outputs.consistency_loss
                
            # else: ## 新增，让一致性损失从一开始就加入，但是随answer_reason重心转移而慢慢预热，也就是权重从0 慢慢升高到0.1
            #     total_loss += 0.1*answer_weight*outputs.consistency_loss
                

            
            # 生成二分类标签
            Binary_lables = []
            for label in answers:
                if label.lower().startswith('a'):
                    Binary_lables.append(1)  # 正样本
                else:
                    Binary_lables.append(0)  # 负样本
            # 转换为 PyTorch 张量
            Binary_lables = torch.tensor(Binary_lables, dtype=torch.long).to(device)

            logits_list = outputs.classification_logits_list
            ### 这里loss_list列表的使用还可以优化
            for i,logits in enumerate(logits_list):
                if logits is not None:
                    torch.cuda.synchronize()
                    if i == 0:
                        temp_loss0 = criterion(logits,Binary_lables) # 计算二分类的二值交叉熵损失, image query
                        total_loss += 0.1*temp_loss0 ##给二分类加个权
                        loss_list.append(temp_loss0)
                        
                    if i == 1: #是output_coord
                        output_coords = logits.to(device)
                        tensor_fake_image_box = torch.cat(fake_image_box, dim=0).reshape(len(fake_image_box), -1).to(device)
                        loss_bbox, loss_giou = get_bbox_loss(output_coords, tensor_fake_image_box) ## output_coords是归一化后的坐标
                        total_loss += 0.1*(loss_bbox+loss_giou) ##给坐标损失加个权
                        loss_list.append(loss_bbox)
                        loss_list.append(loss_giou)
                else:
                    print(f'Attention!! !!模型返回的logits[{i}]是None!!! ')

            total_loss = total_loss / accumulation_steps            
            # -------------------------------------------
            # 前 (accumulation_steps - 1) 次不进行梯度同步
            # DDP 训练中，默认每次backward()都会进行多卡通讯，为下一步的optimizer做准备
            # 但是使用梯度累积的话，没必要每次backward都通讯，白白浪费时间
            # -------------------------------------------
            if (step + 1) % accumulation_steps != 0:
                with model.no_sync():
                    total_loss.backward()
            else:
                total_loss.backward()
                # 梯度裁剪（可选）
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
                if outputs.cos_sim_list:
                    wandb.log({"step": global_step + 1, "cos_sim_mean": outputs.cos_sim_list[0].item()})
                    wandb.log({"step": global_step + 1, "cos_sim_min": outputs.cos_sim_list[1].item()})
                    
                wandb.log({"step": global_step + 1, "step_weighted_sum_loss": total_loss.item()})
                wandb.log({"step": global_step + 1, "step_avg_answer_loss": outputs.loss.item()})
                wandb.log({"step": global_step + 1, "step_avg_reason_loss": outputs.reason_loss.item()})
                wandb.log({"step": global_step + 1, "step_avg_consistency_loss": outputs.consistency_loss.item()})
                
                
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
                                                # 创建并保存 best_info.txt 文件
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
            
            # save_model(model, processor, output_dir, optimizer=optimizer, lr_scheduler=lr_scheduler, epoch=epoch)
            save_model(model, processor, output_dir, optimizer=None, lr_scheduler=None, epoch=epoch) ##不保存优化器状态
            
            

    # Finish the wandb run
    if rank == 0 and need_wandb:
        wandb.finish()

    cleanup()


def main(train_time,local_train_name):
    global train_js, val_js_list, dataset_root, florence_init_pth, florence_base_path
    global output_root, need_replace_lmHead, need_wandb, logged_task_name
    global num_workers, consist_warm_step, reason_answer_warmup

    parser = argparse.ArgumentParser(description="Stage 2: Reasoning-Endowed Joint Fine-Tuning")
    parser.add_argument("--dataset", type=str, default="ROM", choices=["ROM"], help="Dataset to train on")
    parser.add_argument("--train-json", required=True, help="ROM training meta.json")
    parser.add_argument("--val-json", nargs="+", required=True, help="One or more ROM validation meta.json files")
    parser.add_argument("--dataset-root", default=None, help="Root of REFORM_ROMdataset for relative image paths")
    parser.add_argument("--model-path", required=True, help="Stage-1 checkpoint, normally epoch_4")
    parser.add_argument("--florence-base-path", default=None, help="Base Florence-2 checkpoint used to restore lm_head")
    parser.add_argument("--output-dir", default=output_root, help="Directory for checkpoints and logs")
    parser.add_argument("--task-name", default=logged_task_name)
    parser.add_argument("--batch-size", type=int, default=train_bs, help="Batch size for training") ## batch_size设置成5刚好跑到23G显存左右，这个batchsize指的是一张卡的size
    parser.add_argument("--use-lora", action='store_true', help="Use LoRA if this flag is passed")
    parser.add_argument("--epochs", type=int, default=train_epoch, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=main_lr, help="Learning rate")
    parser.add_argument("--eval-steps", type=int, default=4000, help="Number of steps between evaluations, 4kstep, when bs is 4, is about 50min")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for wandb")
    parser.add_argument("--max-val-item-count", type=int, default=4900, help="Maximum number of items to evaluate on during validation")
    parser.add_argument("--replace-lm-head", action=argparse.BooleanOptionalAction, default=need_replace_lmHead)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-base-url", default=None)
    parser.add_argument("--num-workers", type=int, default=num_workers)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--consist-warm-step", type=float, default=consist_warm_step)
    parser.add_argument("--reason-answer-warmup", type=float, default=reason_answer_warmup)
    args = parser.parse_args()

    train_js = args.train_json
    val_js_list = args.val_json
    dataset_root = args.dataset_root
    florence_init_pth = args.model_path
    florence_base_path = args.florence_base_path
    output_root = args.output_dir
    need_replace_lmHead = args.replace_lm_head
    need_wandb = args.wandb
    logged_task_name = args.task_name
    local_train_name = args.run_name or f"{logged_task_name}_{train_time}"
    num_workers = args.num_workers
    consist_warm_step = args.consist_warm_step
    reason_answer_warmup = args.reason_answer_warmup
    if args.wandb_base_url:
        os.environ["WANDB_BASE_URL"] = args.wandb_base_url
    os.environ["REFORM_TRAIN_JSON"] = train_js
    os.environ["REFORM_VAL_JSONS"] = os.pathsep.join(val_js_list)
    if dataset_root:
        os.environ["REFORM_DATASET_ROOT"] = dataset_root
    os.environ["REFORM_MODEL_PATH"] = florence_init_pth
    if florence_base_path:
        os.environ["REFORM_FLORENCE_BASE_PATH"] = florence_base_path
    os.environ["REFORM_OUTPUT_ROOT"] = output_root
    os.environ["REFORM_TASK_NAME"] = logged_task_name
    os.environ["REFORM_REPLACE_LM_HEAD"] = str(need_replace_lmHead)
    os.environ["REFORM_WANDB"] = str(need_wandb)
    os.environ["REFORM_NUM_WORKERS"] = str(num_workers)
    os.environ["REFORM_CONSIST_WARM_STEP"] = str(consist_warm_step)
    os.environ["REFORM_REASON_ANSWER_WARMUP"] = str(reason_answer_warmup)
    
    ## 几个GPU就用几个world_size
    world_size = args.world_size or torch.cuda.device_count()
    # world_size = 1
    
    mp.spawn(
        train_model,
        args=(world_size, args.dataset, train_time, args.batch_size, args.use_lora, args.epochs, args.lr, args.eval_steps, local_train_name, args.max_val_item_count),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    
    train_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_train_name = f'{logged_task_name}_{train_time}' ##wandb上的任务名 
    
    main(train_time,local_train_name)
