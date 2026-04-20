# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

RANK = int(os.getenv("LOCAL_RANK", 0))
DEBUG = os.environ.get("ZYC_DEBUG_MODE", "false").lower() == "true"

# 2) DataLoader 工作者数强制为 0（如果你自己构建 DataLoader）
# dataloader = DataLoader(dataset, batch_size=..., num_workers=0 if DEBUG else 4)

import sys
import re
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reform.rl_trainer import REFORMNoSelfRewardGRPOTrainer as REFORMBERTRewardGRPOTrainer


from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from reform.grpo_config import GRPOConfig
import PIL


import json
import math

from typing import Tuple
from transformers.utils import logging

from PIL import Image
import torch

logger = logging.get_logger(__name__)

tokenizer = None


# GRPO训练参数
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    test_data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to evaluation data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )    
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )



def extract_bbox(response):
    pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
    match = re.search(pattern, response)
    # print(f'input text is {text}')
    
    if match:
        # 将匹配到的坐标转换为整数
        loc_x1 = int(match.group(1))
        loc_y1 = int(match.group(2))
        loc_x2 = int(match.group(3))
        loc_y2 = int(match.group(4))
        # print('解析到的坐标是：')
        # print(loc_x1, loc_y1, loc_x2, loc_y2)
        return [loc_x1, loc_y1, loc_x2, loc_y2], True
    else:
        # print('没有match')
        return [0, 0, 0, 0], False


def extract_choice(response):
    # 提取选项
    choice = response.split('Fake area')[0].strip()

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        # local_rank = int(os.getenv("LOCAL_RANK", 0))
        with open(log_path, "a") as f:
            f.write(f"-------------extract_choice-------------\n")
            f.write(f"response: {response}\n")
            f.write(f"extracte_choice: {choice}\n")

    return choice.strip()


def accuracy_reward_choice(completions, solution, **kwargs):
    """
    判断选择题是否一致
    """

    contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    debug_mode = os.getenv("DEBUG_MODE") == "true"
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol in zip(contents, solution):
        reward = 0.0
        student_answer_action = None
        ground_truth_action = None
        try:
            student_answer_action = (extract_choice(content) or "").lower()
            ground_truth_action = (extract_choice(sol) or "").lower()

            if student_answer_action == ground_truth_action and student_answer_action != "":
                reward = 1.0
            elif student_answer_action[:2] == ground_truth_action[:2] and student_answer_action != "":
                reward = 0.5
            else:
                reward = 0.0

        except Exception as e:
            reward = 0.0

        rewards.append(reward)

        if debug_mode and RANK == 0:
            log_path = os.getenv("LOG_PATH", "reward_debug.log")
            with open(log_path, "a") as f:
                f.write(f"\n--- {current_time} accuracy_reward_choice ---\n")
                f.write(f"content: {content}\n")
                f.write(f"solution: {sol}\n")
                f.write(f"student_answer_action: {student_answer_action}\n")
                f.write(f"ground_truth_action: {ground_truth_action}\n")
                f.write(f"reward: {reward}\n")

    return rewards


def BERT_reward(completions, solution, **kwargs):
    """
    判断选择题是否一致
    IMAGE_TYPES = ["orig", "face_swap", "face_attribute", "full_gene", "bg_rep"] ## 对应 0，1，2，3，4
    TEXT_TYPES = ["orig", "rewritten"] ## 对应 0，1
    """

    contents = [completion[0]["content"] for completion in completions]
    fake_cls_values = kwargs.get("fake_cls", [""] * len(contents))
    
    rewards = []
    debug_mode = os.getenv("DEBUG_MODE") == "true"
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol, fake_cls in zip(contents, solution, fake_cls_values):
        reward = 0.0

        try:
            student_answer_option = (extract_choice(content) or "").lower()[0]
            ground_truth_action = (extract_choice(sol) or "").lower()[0]
            text_label = "rewritten" if "text_swap" in fake_cls else "orig"
            if fake_cls == "orig" or fake_cls == "text_swap":
                image_label = "orig"
            else:
                image_label = fake_cls.replace("&text_swap", "")
            
            ## 如果answer和label不一致，那么reason输出也认为是没有意义的
            if ground_truth_action != student_answer_option or student_answer_option=="": 
                reward = 0.0
            ## 如果answer回答对了，再处理reason输出的奖励
            else: 
                ## text模态
                if text_label == 'orig' and student_answer_option in ['a', 'b', 'c', 'd', 'e']:
                    reward += 0.5
                elif text_label != 'orig' and student_answer_option in ['f', 'g', 'h', 'i', 'j']:
                    reward += 0.5
                
                ## image 模态
                if image_label == 'orig' and student_answer_option in ['a', 'j']:
                    reward += 0.5
                elif image_label == 'face_swap' and student_answer_option in ['b', 'f']:
                    reward += 0.5
                elif image_label == 'face_attribute' and student_answer_option in ['c', 'g']:
                    reward += 0.5
                elif image_label == 'full_gene' and student_answer_option in ['d', 'h']:
                    reward += 0.5
                elif image_label == 'bg_rep' and student_answer_option in ['e', 'i']:
                    reward += 0.5
                
        except Exception as e:
            reward = 0.0

        rewards.append(reward)

        if debug_mode and RANK == 0:
            log_path = os.getenv("LOG_PATH", "reward_debug.log")
            with open(log_path, "a") as f:
                f.write(f"\n--- {current_time} BERT_reward ---\n")
                f.write(f"content: {content}\n")
                f.write(f"student_answer_option: {student_answer_option}\n")
                f.write(f"image_label: {image_label}\n")
                f.write(f"text_label: {text_label}\n")
                
    return rewards



def real_fake_reward_choice(completions, solution, **kwargs):
    """
    判断选择题是否一致
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    debug_mode = os.getenv("DEBUG_MODE") == "true"
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")


    for content, sol in zip(contents, solution):
        reward = 0.0
        success = False
        student_answer_action = None
        ground_truth_action = None
        try:
            student_answer_action = (extract_choice(content) or "").lower()
            ground_truth_action = (extract_choice(sol) or "").lower()

            # 真假 判别 因为提取时使用了lower，所以用小写比较
            success = ground_truth_action.startswith("a.") == student_answer_action.startswith("a.")
                 
            if success and student_answer_action != "":
                reward = 1.0

            else:
                reward = 0.0

        except Exception as e:
            reward = 0.0

        rewards.append(reward)

        if debug_mode and RANK == 0:
            log_path = os.getenv("LOG_PATH", "reward_debug.log")
            with open(log_path, "a") as f:
                f.write(f"\n--- {current_time} real_fake_reward_choice ---\n")
                f.write(f"student_answer_action: {student_answer_action}\n")
                f.write(f"ground_truth_action: {ground_truth_action}\n")
                f.write(f"reward: {reward}\n")

    return rewards


def iou(box1, box2):
    """
    Compute IoU between two boxes.
    box1, box2: [x1, y1, x2, y2], assumed to be in the same coordinate scale (pixel or normalized)
    """
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def accuracy_reward_bbox(completions, solution, **kwargs):
    """
    评估BBox或点击类问题的准确性
    """
    forbidden_choices = {"A", "D", "E", "H", "I", "J"}  # 不应含bbox
    required_choices = {"B", "C", "F", "G"}            # 应含bbox

    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    debug_mode = os.getenv("DEBUG_MODE") == "true"
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol in zip(contents, solution):
        reward = 0.0
        student_answer_action = None
        ground_truth_action = None
        student_bbox = None
        ground_truth_bbox = None

        student_answer_action = extract_choice(content)
        ground_truth_action = extract_choice(sol)
        choice = ground_truth_action[0] if ground_truth_action else None

        student_bbox, student_bbox_flag = extract_bbox(content)
        ground_truth_bbox, gt_bbox_flag = extract_bbox(sol)

        if choice in forbidden_choices:
            reward = 0.0 if student_bbox_flag else 1.0
            
        elif choice in required_choices:
            if student_bbox_flag and gt_bbox_flag:
                # IoU 或坐标检测
                reward = iou(student_bbox, ground_truth_bbox)
            else:
                reward = 0.0
        else:
            reward = 0.0


        if debug_mode and RANK == 0:
            log_path = os.getenv("LOG_PATH", "reward_debug.log")
            with open(log_path, "a") as f:
                f.write(f"\n--- {current_time} accuracy_reward_bbox ---\n")
                f.write(f"content: {content}\n")
                f.write(f"solution: {sol}\n")
                f.write(f"student_answer_action: {student_answer_action}\n")
                f.write(f"ground_truth_action: {ground_truth_action}\n")
                f.write(f"student_bbox: {student_bbox}\n")
                f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
                f.write(f"reward: {reward}\n")
                
        rewards.append(reward)
        
    return rewards


def format_reward(completions, **kwargs):
    """
    满足 [A. B. C. D. E. F. G. H. I. J.]开头 
    如果‘Face’在回答中，那么回答中应该有<loc_{int(x1)}><loc_{int(y1)}><loc_{int(x2)}><loc_{int(y2)}>的坐标, extract_bbox(generated_text) 的返回为true

    """
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    debug_mode = os.getenv("DEBUG_MODE") == "true"
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content in completion_contents:
        
        reward = 0.0
        _, has_box = extract_bbox(content)
        # 1. 必须以 [A-J]. 开头
        if re.match(r"^[A-J]\.", content.strip()):
            # 2. 如果包含 "face"，必须解析到坐标
            if "face swap" in content.lower() or "face attribute" in content.lower():
                if has_box:
                    reward = 1.0
            else:
                if has_box:
                    reward = 0.0
                else:
                    reward = 1.0
        ## 放小一点format reward
        rewards.append(reward)
        if debug_mode and RANK == 0:
            log_path = os.getenv("LOG_PATH", "reward_debug.log")
            with open(log_path, "a") as f:
                f.write(f"\n--- {current_time} format_reward ---\n")
                f.write(f"content: {content.strip()}\n")
                f.write(f"reward: {reward}\n")

    return rewards

# 三个reward的定义
# action_type对应的reward
# 坐标对应的reward
# 输出格式对应的reward
###  reward registry three parts
reward_funcs_registry = {
    "accuracy_choice": accuracy_reward_choice,
    "accuracy_bbox": accuracy_reward_bbox,
    "format": format_reward,
    "real_fake_reward":real_fake_reward_choice,
    "BERT_reward":BERT_reward,
    ## 记得 需要修改下面的script_args.reward_funcs，才能真正选择好奖励函数
    ## 并且配置好奖励函数的权重
}

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
    
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
)



def main(script_args, training_args, model_args):
    
        
    V2_describles_answ_10opts = {}
    V2_describles_answ_10opts['orig'] = "A. No." #[0,0,0,0,0]
    V2_describles_answ_10opts['face_swap'] = "B. Image: Face swap; Text: No." #[1,0,0,0,0]
    V2_describles_answ_10opts['face_attribute'] = "C. Image: Face attribute; Text: No."#[0,1,0,0,0]
    V2_describles_answ_10opts['full_gene'] = "D. Image: Whole generated; Text: No." #[0,0,1,0,0]
    V2_describles_answ_10opts['bg_rep'] = "E. Image: Inpainted background; Text: No." #[0,0,0,1,0]

    V2_describles_answ_10opts['face_swap&text_swap'] = "F. Image: Face swap; Text: Fully rewritten." #[1,0,0,0,1]
    V2_describles_answ_10opts['face_attribute&text_swap'] = "G. Image: Face attribute; Text: Fully rewritten."#[0,1,0,0,1]
    V2_describles_answ_10opts['full_gene&text_swap'] = "H. Image: Whole generated; Text: Fully rewritten."#[0,0,1,0,1]
    V2_describles_answ_10opts['bg_rep&text_swap'] = "I. Image: Inpainted background; Text: Fully rewritten." #[0,0,0,1,1]

    V2_describles_answ_10opts['text_swap'] = "J. Image: No; Text: Fully rewritten."#[0,0,0,0,1]
    describe_temple = "The following are multiple choice questions about fake news detection.\nThe text caption of the news is: "
    describe_ques_latter_10opts = (
        ".\n The image and text should not be manipulated. Question: Is there any manipulation in the image or text of this news?\n"
        "A. No.\n"
        "B. Image: Face swap; Text: No.\n"
        "C. Image: Face attribute; Text: No.\n"
        "D. Image: Whole generated; Text: No.\n"
        "E. Image: Inpainted background; Text: No.\n"
        "F. Image: Face swap; Text: Fully rewritten.\n"
        "G. Image: Face attribute; Text: Fully rewritten.\n"
        "H. Image: Whole generated; Text: Fully rewritten.\n"
        "I. Image: Inpainted background; Text: Fully rewritten.\n"
        "J. Image: No; Text: Fully rewritten.\n"
                         )
    V2_face_locate = "If the face is manipulated, locate the manipulated face in the image and append the results to your selected option.\nThe answer is:"
    # Get reward functions
    
    ## 权重和reward名必须一一对应
    # reward_weights = [2.0, 1.0, 1.0, 5.0, 1.0] 
    reward_weights = None ## 不定义权重，就默认使用sum
    script_args.reward_funcs = ['accuracy_choice','accuracy_bbox','format', 'real_fake_reward','BERT_reward']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # reward_funcs = reward_funcs = list(reward_funcs_registry.values())


    # Load the dataset from huggingface
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset from local disk
    from datasets import DatasetDict
    # dataset = DatasetDict.load_from_disk(script_args.dataset_name)
    import json
    from datasets import Dataset
    
    data_files = script_args.data_file_paths.split(":")
    test_data_files = script_args.test_data_file_paths.split(":")
    image_roots = []
    if script_args.image_folders and script_args.image_folders != "None":
        image_roots = [pathlib.Path(p) for p in script_args.image_folders.split(":")]
    
    def load_json_dataset(file_paths):
        all_data = []
        for file_idx, data_file in enumerate(file_paths):
            image_root = image_roots[min(file_idx, len(image_roots) - 1)] if image_roots else pathlib.Path(data_file).parent
            with open(data_file, 'r') as f:
                data = json.load(f)
                for item in data:
                    caption = item['text']
                    item['image_path'] = item.get('image_path', item.pop('image', None))
                    if item['image_path'] is not None and not pathlib.Path(item['image_path']).is_absolute():
                        item['image_path'] = str((image_root / item['image_path']).resolve())
                    item['problem'] = describe_temple + caption + describe_ques_latter_10opts + V2_face_locate

                    if 'face' in item['fake_cls'] and item.get('fake_image_box'):
                        x1, y1, x2, y2 = item['fake_image_box']
                        item['solution'] = (
                            f"{V2_describles_answ_10opts[item['fake_cls']]}Fake area"
                            f"<loc_{int(x1)}><loc_{int(y1)}><loc_{int(x2)}><loc_{int(y2)}>"
                        )
                    else:
                        item['solution'] = f"{V2_describles_answ_10opts[item['fake_cls']]}"

                    item.pop('conversations', None)
                    all_data.append(item)
        return Dataset.from_list(all_data)
    
    
    dataset = load_json_dataset(data_files)
    test_dataset = load_json_dataset(test_data_files)

    def make_conversation_from_json(example):
        if 'image_path' in example and example['image_path'] is not None:
            # Don't load image here, just store the path
            return {
                # 'image': PIL.Image.open(example['image_path']),
                'image_path': example['image_path'],  # Store path instead of loaded image
                # 'problem': example['problem'],
                'solution': example['solution'],
                # 'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'text': None},
                        {'type': 'text', 'text': example['problem']}
                    ]
                }]
            }
        else:
            return {
                'problem': example['problem'],
                'solution': example['solution'],
                # 'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': example['problem']}
                    ]
                }]
            }

    dataset = dataset.map(make_conversation_from_json, num_proc=8)
    cross_domain_eva_set = test_dataset.map(make_conversation_from_json, num_proc=8)
    splits = {'train': dataset, 'test':cross_domain_eva_set}
    

    trainer_cls = REFORMBERTRewardGRPOTrainer
    print("using trainer:", trainer_cls.__name__)
    
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # print(f'preds is {preds};  labels is {labels}')
        # 确保都是小写并去除空格
        preds = [p.strip().lower() for p in preds]
        labels = [l.strip().lower() for l in labels]
        # 计算准确率
        correct = sum(p.startswith('a') == l.startswith('a') for p, l in zip(preds, labels))
        acc = correct / len(labels)
        ## 与下面的training_args.metric_for_best_model要同步
        return {"eval_accuracy": acc}

    
    ## eva版的trainer，还没改通
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        reward_weights = reward_weights, ## 奖励函数的权重设置
        args=training_args,
        train_dataset=splits['train'],
        eval_dataset = splits['test'],
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        torch_dtype="bfloat16",  # 强制使用bfloat16以匹配模型权重
        compute_metrics=compute_metrics,
        best_metric_name='eval_accuracy'
        
    )
    
    # trainer = trainer_cls(
    #     model=model_args.model_name_or_path,
    #     reward_funcs=reward_funcs,
    #     args=training_args,
    #     train_dataset = splits['train'],
    #     eval_dataset = None,
    #     peft_config=get_peft_config(model_args),
    #     attn_implementation=model_args.attn_implementation,
    #     torch_dtype="bfloat16",  # 强制使用bfloat16以匹配模型权重
    # )
    
    ### 断点重训的问题，因为使用了deepspeed，deepspeed需要恢复的断点格式与HF checkpoint并不一样
    #### 这里不知道为什么保存的是普通的HF断点，所以无法使用deepspeed重训 ***有待解决***
    # # 查找可用 checkpoint（路径中包含 checkpoint）
    # checkpoint_paths = [p for p in pathlib.Path(training_args.output_dir).glob("checkpoint-*") if p.is_dir()]

    # # 选最新的 checkpoint（按数字排序）
    # if checkpoint_paths:
    #     checkpoint_paths.sort(key=lambda x: int(x.name.split("-")[-1]))
    #     latest_checkpoint = checkpoint_paths[-1]
    #     print(f'<-------------启用断点重训: {latest_checkpoint} ---------------->')
    #     trainer.train(resume_from_checkpoint=str(latest_checkpoint))

    # else:
    #     print(f'<-------------不启用断点重训---------------->')
    #     trainer.train()

    trainer.train()
    

    # Save and push to hub
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    training_args.metric_for_best_model='eval_accuracy'
    main(script_args, training_args, model_args)
