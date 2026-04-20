import os
import re
from functools import partial
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.nn.functional as F
from reform import box_ops
import torch
import wandb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from reform.multilabel_metrics import AveragePrecisionMeter
from torchvision.ops.boxes import box_area
import torch.multiprocessing as mp


def box_iou(boxes1, boxes2, test=False):
    '''
    计算两个边界框集合的 IoU（Intersection over Union），
    并返回每个边界框对的 IoU 值和并集面积。
    '''
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    # inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    # union = area1[:, None] + area2 - inter
    union = area1 + area2 - inter

    iou = inter / union

    if test:
        zero_lines = boxes2==torch.zeros_like(boxes2)
        zero_lines_idx = torch.where(zero_lines[:,0]==True)[0]

        for idx in zero_lines_idx:
            if all(boxes1[idx,:] < 1e-4):
                iou[idx]=1

    return iou, union

def collate_fn(batch, processor, device):


    images, questions, answers,fake_image_box, reason, image_paths = zip(*batch)
    
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True)
    # inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device) ##在这里就放入GPU会拉慢训练速度
    
    return inputs, answers, fake_image_box,reason

def create_data_loaders(
    train_dataset,
    val_datasets,
    batch_size,
    num_workers,
    rank,
    world_size,
    processor,
    device,
    eva_batch_size=4,
):
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        sampler=train_sampler,
        
        ##加快CPU -> GPU 数据传输速度的设置
        num_workers=num_workers,
        pin_memory=True,
        
        
    )

    val_loaders = {}
    for name, val_dataset in val_datasets.items():
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        val_loader = DataLoader(
            val_dataset,
            batch_size=eva_batch_size,
            collate_fn=partial(collate_fn, processor=processor, device=device),
            sampler=val_sampler,
    
            ##加快CPU -> GPU 数据传输速度的设置
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loaders[name] = val_loader

    return train_loader, val_loaders

def get_multi_label(answers, device):
    # 初始化 multi_label 矩阵
    multi_label = torch.zeros([len(answers), 5], dtype=torch.long).to(device)

    # 定义 real_label_pos（精确匹配 'A. No.'）
    real_label_pos = [i for i, ans in enumerate(answers) if 'A. No.' in ans]
    multi_label[real_label_pos, :] = torch.tensor([0, 0, 0, 0, 0]).to(device)

    # face_swap cls = [1, 0, 0, 0, 0]
    pos = [i for i, ans in enumerate(answers) if 'B. Image: Face swap; Text: No.' in ans]
    multi_label[pos, :] = torch.tensor([1, 0, 0, 0, 0]).to(device)

    # face_attribute cls = [0, 1, 0, 0, 0]
    pos = [i for i, ans in enumerate(answers) if 'C. Image: Face attribute; Text: No.' in ans]
    multi_label[pos, :] = torch.tensor([0, 1, 0, 0, 0]).to(device)

    # full_gene cls = [0, 0, 1, 0, 0]
    pos = [i for i, ans in enumerate(answers) if 'D. Image: Whole generated; Text: No.' in ans]
    multi_label[pos, :] = torch.tensor([0, 0, 1, 0, 0]).to(device)

    # bg_rep cls = [0, 0, 0, 1, 0]
    pos = [i for i, ans in enumerate(answers) if 'E. Image: Inpainted background; Text: No.' in ans]
    multi_label[pos, :] = torch.tensor([0, 0, 0, 1, 0]).to(device)

    # face_swap & text_swap = [1, 0, 0, 0, 1]
    pos = [i for i, ans in enumerate(answers) if 'F. Image: Face swap; Text: Fully rewritten.' in ans]
    multi_label[pos, :] = torch.tensor([1, 0, 0, 0, 1]).to(device)

    # face_attribute & text_swap = [0, 1, 0, 0, 1]
    pos = [i for i, ans in enumerate(answers) if 'G. Image: Face attribute; Text: Fully rewritten.' in ans]
    multi_label[pos, :] = torch.tensor([0, 1, 0, 0, 1]).to(device)

    # full_gene & text_swap = [0, 0, 1, 0, 1]
    pos = [i for i, ans in enumerate(answers) if 'H. Image: Whole generated; Text: Fully rewritten.' in ans]
    multi_label[pos, :] = torch.tensor([0, 0, 1, 0, 1]).to(device)

    # bg_rep & text_swap = [0, 0, 0, 1, 1]
    pos = [i for i, ans in enumerate(answers) if 'I. Image: Inpainted background; Text: Fully rewritten.' in ans]
    multi_label[pos, :] = torch.tensor([0, 0, 0, 1, 1]).to(device)

    # only text_swap = [0, 0, 0, 0, 1]
    pos = [i for i, ans in enumerate(answers) if 'J. Image: No; Text: Fully rewritten.' in ans]
    multi_label[pos, :] = torch.tensor([0, 0, 0, 0, 1]).to(device)

    return multi_label, real_label_pos

def get_multi_label_dgm4(answers, device):
    # 初始化 multi_label 矩阵
    multi_label = torch.zeros([len(answers), 4], dtype=torch.long).to(device)
    
    # 定义 real_label_pos（精确匹配 'A. No.'）
    real_label_pos = [i for i, ans in enumerate(answers) if 'A. No.' in ans ]
    multi_label[real_label_pos, :] = torch.tensor([0, 0, 0, 0]).to(device)
    
    # face_swap cls = [1, 0, 0, 0]（精确匹配 'B. Only face swap.'）
    pos = [i for i, ans in enumerate(answers) if 'B. Only face swap.' in ans ]
    multi_label[pos, :] = torch.tensor([1, 0, 0, 0]).to(device)
    
    # face_attribute cls = [0, 1, 0, 0]（精确匹配 'C. Only face attribute.'）
    pos = [i for i, ans in enumerate(answers) if 'C. Only face attribute.' in ans ]
    multi_label[pos, :] = torch.tensor([0, 1, 0, 0]).to(device)
    
    # text_swap cls = [0, 0, 1, 0]（精确匹配 'D. Only text swap.'）
    pos = [i for i, ans in enumerate(answers) if 'D. Only text swap.' in ans ]
    multi_label[pos, :] = torch.tensor([0, 0, 1, 0]).to(device)
    
    # face_swap&text_swap cls = [1, 0, 1, 0]（精确匹配 'E. Face swap and text swap.'）
    pos = [i for i, ans in enumerate(answers) if 'E. Face swap and text swap.' in ans ]
    multi_label[pos, :] = torch.tensor([1, 0, 1, 0]).to(device)
    
    # face_attribute&text_swap cls = [0, 1, 1, 0]（精确匹配 'F. Face attribute and text swap.'）
    pos = [i for i, ans in enumerate(answers) if 'F. Face attribute and text swap.' in ans ]
    multi_label[pos, :] = torch.tensor([0, 1, 1, 0]).to(device)
    
    return multi_label, real_label_pos

def get_best_option(generated_texts, option_vectors,vectorizer,options,option_labels,device):
    '''批量计算模型的输出对应哪一个选项
    输入是生成的多个文本，和固定选项的向量表示
    '''
    # 将生成文本批量转换为向量
    generated_vectors = vectorizer.transform(generated_texts).toarray()

    # 计算相似度
    similarities = cosine_similarity(generated_vectors, option_vectors)

    # 获取每个生成文本的相似度最高的选项
    best_option_indices = similarities.argmax(axis=1)

    # 返回选项、相似度和对应的01标签
    best_options = [options[i] for i in best_option_indices]
    best_similarities = [similarities[i, best_option_indices[i]] for i in range(len(generated_texts))]

    best_multi_labels = torch.stack([option_labels[i] for i in best_option_indices], dim=0)
    # 对 best_multi_labels 进行归一化
    # best_multi_labels_prob = F.softmax(best_multi_labels.float(), dim=1)
    
    #ori_pos，构造模型输出对应的单分类标签
    pred_label = torch.ones(len(generated_texts), dtype=torch.long).to(device) 
    real_label_pos = np.where(np.array(best_options) == 'A. No.')[0].tolist()
    # 是A. No.的地方设置为 0 --代表real图文
    pred_label[real_label_pos] = 0
    
    return best_options, best_similarities, best_multi_labels,pred_label

def synchronize_metrics(metric_tensor, world_size):
    """
    使用reduce操作同步指标。聚合各个进程的指标，计算全局值。
    """
    # 将指标结果归约到 rank 0 进程
    dist.reduce(metric_tensor, dst=0, op=dist.ReduceOp.SUM)
    # 在 rank 0 进程计算均值
    if dist.get_rank() == 0:
        metric_tensor /= world_size
    return metric_tensor

def parse_coordinates(text):
    # 使用正则表达式匹配坐标
    pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
    match = re.search(pattern, text)
    # print(f'input text is {text}')
    
    if match:
        # 将匹配到的坐标转换为整数
        loc_x1 = int(match.group(1))
        loc_y1 = int(match.group(2))
        loc_x2 = int(match.group(3))
        loc_y2 = int(match.group(4))
        # print('解析到的坐标是：')
        # print(loc_x1, loc_y1, loc_x2, loc_y2)
        return torch.tensor([[loc_x1, loc_y1, loc_x2, loc_y2]])
    else:
        # print('没有match')
        return torch.tensor([[0, 0, 0, 0]])

def get_bbox_loss(output_coord, target_bbox, is_image=None):
    """
    Bounding Box Loss: L1 & GIoU

    Args:
        image_embeds: encoding full images
    """
    loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4

    boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
    boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
    if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
        # early check of degenerated boxes
        # print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
        loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
    else:
        # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz
        loss_giou = 1 - box_ops.generalized_box_iou(boxes1, boxes2)  # bsz

    if is_image is None:
        num_boxes = target_bbox.size(0)
    else:
        num_boxes = torch.sum(1 - is_image)
        loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
        loss_giou = loss_giou * (1 - is_image)

    return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes




def evaluate_model(rank, world_size, model, val_loaders, device, processor, global_step, max_val_item_count,option_vectors,vectorizer,options,option_labels, need_wandb=True):

    model.eval()
    with torch.no_grad():
        for val_name, val_loader in val_loaders.items():
            val_item_count = 0
            cls_nums_all = 0
            cls_acc_all = 0 
            IOU_pred = []
            multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
            multi_label_meter.reset()
            for batch in tqdm(val_loader, desc=f"Evaluation on {val_name} at step {global_step}", position=rank):
                # inputs, batch_answers = batch
                inputs, batch_answers, fake_image_box, reason  = batch
                val_item_count += len(batch_answers)
                ## model是DistributedDataParallel包装后的类，并没有generate方法，如需调用，应该使用model.module调用基础模型后再调用generate()方法
                
                input_ids = inputs["input_ids"].to(device)
                pixel_values = inputs["pixel_values"].to(device)
                
                generated_ids = model.module.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=256,
                    num_beams=3,
                )
                ###解析得到模型的文本输出
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
                
                task_answers = []
                
                output_coords = torch.zeros((len(generated_texts), 4)).to(device)
                true_coords = torch.zeros((len(generated_texts), 4)).to(device)
                
                for i, (generated_text, answers) in enumerate(zip(generated_texts, batch_answers)):

                    full_answer = re.sub(r"<pad>|<s>|</s>", "", generated_text)
                    
                    
                    if '<loc_' in full_answer:
                        task_answers.append(full_answer.split('Fake area')[0])
                        output_coords[i] = parse_coordinates(full_answer).to(device)
                        true_coords[i] = parse_coordinates(answers).to(device)
    
                    # 将 output_coord 堆叠到 output_coords中
                    else:
                        task_answers.append(full_answer)
                        true_coords[i] = parse_coordinates(answers).to(device)
                
                real_multi_label, real_label_pos = get_multi_label(batch_answers,device)
                real_label = torch.ones(len(generated_texts), dtype=torch.long).to(device) 
                real_label[real_label_pos] = 0
                best_options, _ , best_multi_labels, pred_label = get_best_option(task_answers, option_vectors,vectorizer,options,option_labels,device)
                
                ##--reeal/fake---##
                cls_nums_all = val_item_count
                cls_acc_all += torch.sum(real_label == pred_label).item()
                
                # ##-IoU--##
                # IOU, _ = box_iou(output_coords, true_coords.to(device), test=True)
                # for iou_value in IOU.cpu().tolist():
                #     if isinstance(iou_value, (int, float)) and not math.isnan(iou_value) and not math.isinf(iou_value):
                #         IOU_pred.append(iou_value)
                #     else:
                #         IOU_pred.append(0.0)
        ######################################

            
                ##-multi--##
                # multi_label_meter.add(best_multi_labels, real_multi_label)
                                
                
                # 计算本进程的各项指标
                local_ACC_cls = cls_acc_all / cls_nums_all
                # local_IOU_score = sum(IOU_pred)/len(IOU_pred)
                # local_MAP = multi_label_meter.value()[:3].mean().item()


                if val_item_count > max_val_item_count:
                    break
        # 同步各进程计算的指标
        local_ACC_cls_tensor = torch.tensor(local_ACC_cls, device=device)
        # local_IoU_score_tensor = torch.tensor(local_IOU_score, device=device)
        # local_MAP_tensor = torch.tensor(local_MAP, device=device)
        


        # 聚合指标到主进程
        ACC_cls = synchronize_metrics(local_ACC_cls_tensor, world_size)
        # IoUscore = synchronize_metrics(local_IoU_score_tensor, world_size)
        # MAP = synchronize_metrics(local_MAP_tensor, world_size)
        


        # 打印和记录日志
        if dist.get_rank() == 0 and need_wandb:
            print(f"Rank {rank} - Step {global_step} - ACC perform ({val_name}): {ACC_cls.item()}")
            wandb.log({
                f"eva_ACC_cls": ACC_cls.item(),
                # f"{val_name}_IoUscore": IoUscore.item(),
                # f"{val_name}_MAP": MAP.item(),
                
                "step": global_step
            })
            
    model.train()
    return ACC_cls




def evaluate_model_dgm4(rank, world_size, model, val_loaders, device, processor, global_step, max_val_item_count,option_vectors,vectorizer,options,option_labels, need_wandb=True):

    model.eval()
    with torch.no_grad():
        for val_name, val_loader in val_loaders.items():
            val_item_count = 0
            cls_nums_all = 0
            cls_acc_all = 0 
            IOU_pred = []
            multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
            multi_label_meter.reset()
            for batch in tqdm(val_loader, desc=f"Evaluation on {val_name} at step {global_step}", position=rank):
                # inputs, batch_answers = batch
                inputs, batch_answers, fake_image_box, reason  = batch
                val_item_count += len(batch_answers)
                ## model是DistributedDataParallel包装后的类，并没有generate方法，如需调用，应该使用model.module调用基础模型后再调用generate()方法
                generated_ids = model.module.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    num_beams=3,
                )
                ###解析得到模型的文本输出
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
                
                task_answers = []
                
                output_coords = torch.zeros((len(generated_texts), 4)).to(device)
                true_coords = torch.zeros((len(generated_texts), 4)).to(device)
                
                for i, (generated_text, answers) in enumerate(zip(generated_texts, batch_answers)):

                    full_answer = re.sub(r"<pad>|<s>|</s>", "", generated_text)
                    
                    
                    if '<loc_' in full_answer:
                        task_answers.append(full_answer.split('Fake area')[0])
                        output_coords[i] = parse_coordinates(full_answer).to(device)
                        true_coords[i] = parse_coordinates(answers).to(device)
    
                    # 将 output_coord 堆叠到 output_coords中
                    else:
                        task_answers.append(full_answer)
                        true_coords[i] = parse_coordinates(answers).to(device)
                
                real_multi_label, real_label_pos = get_multi_label_dgm4(batch_answers,device)
                real_label = torch.ones(len(generated_texts), dtype=torch.long).to(device) 
                real_label[real_label_pos] = 0
                best_options, _ , best_multi_labels, pred_label = get_best_option(task_answers, option_vectors,vectorizer,options,option_labels,device)
                
                ##--reeal/fake---##
                cls_nums_all = val_item_count
                cls_acc_all += torch.sum(real_label == pred_label).item()
                
                # ##-IoU--##
                # IOU, _ = box_iou(output_coords, true_coords.to(device), test=True)
                # for iou_value in IOU.cpu().tolist():
                #     if isinstance(iou_value, (int, float)) and not math.isnan(iou_value) and not math.isinf(iou_value):
                #         IOU_pred.append(iou_value)
                #     else:
                #         IOU_pred.append(0.0)
        ######################################

            
                ##-multi--##
                # multi_label_meter.add(best_multi_labels, real_multi_label)
                                
                
                # 计算本进程的各项指标
                local_ACC_cls = cls_acc_all / cls_nums_all
                # local_IOU_score = sum(IOU_pred)/len(IOU_pred)
                # local_MAP = multi_label_meter.value()[:3].mean().item()


                if val_item_count > max_val_item_count:
                    break
        # 同步各进程计算的指标
        local_ACC_cls_tensor = torch.tensor(local_ACC_cls, device=device)
        # local_IoU_score_tensor = torch.tensor(local_IOU_score, device=device)
        # local_MAP_tensor = torch.tensor(local_MAP, device=device)
        


        # 聚合指标到主进程
        ACC_cls = synchronize_metrics(local_ACC_cls_tensor, world_size)
        # IoUscore = synchronize_metrics(local_IoU_score_tensor, world_size)
        # MAP = synchronize_metrics(local_MAP_tensor, world_size)
        


        # 打印和记录日志
        if dist.get_rank() == 0 and need_wandb:
            print(f"Rank {rank} - Step {global_step} - ACC perform ({val_name}): {ACC_cls.item()}")
            wandb.log({
                f"eva_ACC_cls": ACC_cls.item(),
                # f"{val_name}_IoUscore": IoUscore.item(),
                # f"{val_name}_MAP": MAP.item(),
                
                "step": global_step
            })
            
    model.train()
    return ACC_cls



def save_model(model, processor, output_dir, optimizer=None, lr_scheduler=None, epoch=None):
    
    model_to_save = model.module if hasattr(model, "module") else model
    # 更新 config vocab_size
    model_to_save.config.vocab_size = len(processor.tokenizer)
    model_to_save.config.vision_config.model_type = "davit"
    
    if not hasattr(model_to_save.config, "reason_decoder"):
        model_to_save.config.reason_decoder = {}

        
    # 保存模型和 config
    model_to_save.save_pretrained(output_dir, safe_serialization=False)
    model_to_save.config.save_pretrained(output_dir)

    # 保存 processor / tokenizer
    processor.save_pretrained(output_dir)
                
    if optimizer is not None and lr_scheduler is not None and epoch is not None:
        torch.save({
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, os.path.join(output_dir, "optimizer.pt"))



@torch.no_grad()
def inspect_and_break_sharing(model, verbose=True):
    """
    打断 decoder 与 decoder_2 之间的参数共享，并复制参数。
    同时将 lm_head 的参数复制给 lm_think_head（保持两者独立）。
    """

    decoder = model.language_model.model.decoder
    decoder_2 = model.language_model.model.decoder_2
    
    # 主、辅助输出层
    lm_head = model.language_model.lm_head
    reason_lm = model.language_model.lm_think_head

    src_params = dict(decoder.named_parameters())
    dst_params = dict(decoder_2.named_parameters())

    if verbose:
        print(f"🔍 开始检查并打断共享，共发现 {len(src_params)} 个 decoder 参数。")

    num_shared = 0
    num_copied = 0

    for name, src_param in src_params.items():
        if name not in dst_params:
            if verbose:
                print(f"⚠️ decoder_2 缺少参数: {name}")
            continue

        dst_param = dst_params[name]

        # 检查是否共享同一个 storage
        if id(src_param.data.storage()) == id(dst_param.data.storage()):
            num_shared += 1
            dst_param.data = src_param.data.clone()
            num_copied += 1

            if verbose:
                print(f"🟢 共享已打断并复制参数: {name}")
        else:
            if verbose:
                print(f"✅ 参数 {name} 已独立（未共享）")

    # ====== lm_head -> lm_think_head ======
    if hasattr(lm_head, "weight") and hasattr(reason_lm, "weight"):
        if id(lm_head.weight.data.storage()) == id(reason_lm.weight.data.storage()):
            if verbose:
                print(f"\n🔁 检测到 lm_head 与 lm_think_head 权重共享，正在打断...")
            reason_lm.weight.data = lm_head.weight.data.clone()
        else:
            # 即使未共享，也强制复制初始化，以确保初始化完全一致
            reason_lm.weight.data.copy_(lm_head.weight.data)
        if verbose:
            print("✅ 已将 lm_head 权重复制至 lm_think_head，并确保独立。")

    # ====== 如果存在 bias，也一并复制 ======
    if hasattr(lm_head, "bias") and lm_head.bias is not None:
        if not hasattr(reason_lm, "bias"):
            reason_lm.bias = torch.nn.Parameter(torch.zeros_like(lm_head.bias))
        reason_lm.bias.data.copy_(lm_head.bias.data)
        if verbose:
            print("✅ 已复制 lm_head.bias → lm_think_head.bias。")

    # ====== 汇总信息 ======
    print(f"\n✅ 共享检查完成：发现共享 {num_shared} 个参数，已复制 {num_copied} 个。")
    print(f"✅ lm_head → lm_think_head 权重复制完成。")

    # ====== 打印 reason_lm 的统计信息 ======
    if verbose:
        w = reason_lm.weight.data
        print(f"\n📊 think_lm 权重统计：mean={w.mean().item():.6f}, std={w.std().item():.6f}, "
              f"shape={tuple(w.shape)}")
    
    return model


    # # -------------------------------------------------------------------------
    # # 复制完成后输出 decoder_2 每个参数的形状、均值、标准差和零值占比
    # # -------------------------------------------------------------------------
    # print("\n📊 decoder_2 参数统计信息：")
    # with torch.no_grad():
    #     for name, param in decoder_2.named_parameters():
    #         tensor = param.data.cpu()
    #         mean_val = tensor.float().mean().item()
    #         std_val = tensor.float().std().item()
    #         num_zeros = torch.sum(tensor == 0).item()
    #         total_elements = tensor.numel()
    #         zero_percentage = (num_zeros / total_elements) * 100 if total_elements > 0 else 0

    #         print(f"\n参数名称: {name}")
    #         print(f"  - 形状: {list(tensor.shape)}")
    #         print(f"  - 均值 (mean): {mean_val:.6f}")
    #         print(f"  - 标准差 (std): {std_val:.6f}")
    #         print(f"  - 零值占比: {zero_percentage:.2f}% ({num_zeros}/{total_elements})")

    # print("\n✅ inspect_and_break_sharing 完成。\n")

@torch.no_grad()
def Expand_embed_positions(model, new_max_pos=1236):
    import torch.nn as nn
    #    2. 定位 encoder & decoder 端的原生 Florence2LearnedPositionalEmbedding 实例
    ## 第一次训练florence2 -- 2decoder 的时候，没有把decoder_2也扩展成1236长度的，虽然结果来看，性能变好了
    ## 但是可以尝试一下同样将decoder2也扩展
    device = model.device
    pe_enc = model.language_model.model.encoder.embed_positions
    pe_dec = model.language_model.model.decoder.embed_positions
    pe_dec2 = model.language_model.model.decoder_2.embed_positions
    
    # 3. 直接拿到它们的“类”，不用额外 import 文件
    PosEmbedClass = pe_enc.__class__

    # 4. 读取旧权重信息
    old_enc_w = pe_enc.weight.data.clone()
    old_dec_w = pe_dec.weight.data.clone()
    old_dec2_w = pe_dec2.weight.data.clone()
    
    old_enc_n, dim = old_enc_w.shape
    old_dec_n, _   = old_dec_w.shape
    old_dec2_n, _   = old_dec2_w.shape
    
    offset        = pe_enc.offset   # 两者相同

    new_n         = new_max_pos + offset

    # 6. 用原生的类来实例化新表，并拷贝 + 初始化
    def expand(pe_old_w, old_n):
        new_pe = PosEmbedClass(new_max_pos, dim)
        new_pe.offset = offset
        with torch.no_grad():
            # 拷贝预训练好的前 old_n 行
            new_pe.weight[:old_n].copy_(pe_old_w)
            # 随机初始化新增部分
            nn.init.normal_(
                new_pe.weight[old_n:],
                mean=0.0,
                std=getattr(model.config, "initializer_range", 0.02)
            )
        return new_pe

    new_pe_enc = expand(old_enc_w, old_enc_n).to(device)
    new_pe_dec = expand(old_dec_w, old_dec_n).to(device)
    new_pe_dec2 = expand(old_dec2_w, old_dec2_n).to(device)
    

    # 7. 替换回模型
    model.language_model.model.encoder.embed_positions = new_pe_enc
    model.language_model.model.decoder.embed_positions = new_pe_dec
    model.language_model.model.decoder_2.embed_positions = new_pe_dec2
    
    return model
    
