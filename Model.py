from typing import Any, Optional  # 导入类型注释
import pytorch_lightning as pl  # 导入PyTorch Lightning库
from pytorch_lightning.utilities.types import STEP_OUTPUT  # 导入STEP_OUTPUT类型，用于定义训练步骤输出
from transformers import PreTrainedTokenizer  # 导入预训练tokenizer

import torch  # 导入PyTorch库
import os  # 导入os模块
import json  # 导入json模块
from evaluation_util import calculate_metrics  # 导入自定义评估函数


class Model(pl.LightningModule):  # 定义一个继承自LightningModule的模型类
    def __init__(self, args, model, tokenizer: PreTrainedTokenizer, use_cpu_offload=False,
                 truncate_early=True, max_length=32):
        """
        初始化模型类，参数说明：
        - completion_metadata: 保存完成的元数据。如果为None，则不保存完成数据。
        - model: 传入的模型
        - tokenizer: 用于处理输入文本的tokenizer
        - use_cpu_offload: 是否使用CPU进行卸载
        - truncate_early: 是否提前截断输入
        - max_length: 生成文本的最大长度
        """
        super().__init__()  # 调用父类初始化方法
        self.save_hyperparameters(ignore=['model'])  # 保存超参数，忽略'model'参数
        self.model = model  # 赋值模型
        self.tokenizer = tokenizer  # 赋值tokenizer
        self.use_cpu_offload = use_cpu_offload  # 是否使用CPU卸载
        self.lr = args.lr  # 学习率
        self.max_length = max_length  # 最大长度
        self.truncate_early = truncate_early  # 是否提前截断
        self.outputs = []  # 初始化输出列表
        self.args = args  # 赋值args参数

        # 读取验证集的gold标签数据
        self.gold = []
        with open('data/{}_drr_{}_{}.txt'.format(self.args.dataset, self.args.val_set, self.args.role), 'r') as f:
            lines = f.readlines()  # 读取文件中的每一行
            self.gold = [json.loads(line)['output'] for line in lines]  # 从每一行中提取output字段作为gold标签

    def training_step(self, batch, batch_idx):
        """
        定义训练步骤：
        - 接收批次数据
        - 计算损失并返回
        """
        kwargs = {
            "input_ids": batch["input_ids"],  # 输入ID
            "attention_mask": batch["attention_mask"],  # 注意力掩码
            "labels": batch["labels"],  # 标签
        }

        kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]  # 解码器的注意力掩码
        loss = self.model(**kwargs)["loss"]  # T5 模型是 encoder-decoder 结构，它需要 decoder_input_ids 或 decoder_inputs_embeds 才能做 forward 推理或训练。
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)  # 记录训练损失
        return loss  # 返回损失

    def validation_step(self, batch, batch_idx):
        """
        定义验证步骤：
        - 计算损失
        - 生成预测并保存
        """
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

        kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        loss = self.model(**kwargs)["loss"]  # 计算损失
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)  # 记录验证损失

        pred = self.model.generate(batch["input_ids"], max_length=self.max_length)  # 使用模型生成预测文本

        output = {
            "pred": pred  # 存储预测结果
        }
        self.outputs.append(output)  # 将预测结果添加到outputs列表
        return output  # 返回预测结果

    def on_validation_epoch_end(self) -> None:
        """
        每个验证周期结束时，计算并记录评估指标（如准确率和F1分数）
        """
        pred_strs = []  # 存储解码后的预测文本
        for batch_output in self.outputs:
            batch_decode = self.tokenizer.batch_decode(batch_output['pred'], skip_special_tokens=True)  # 解码预测结果
            pred_strs.extend(batch_decode)  # 扩展到pred_strs列表中

        # 计算自定义指标（准确率和F1分数）
        custom_metrics = calculate_metrics(pred_strs, self.gold)
        self.log("em", custom_metrics['em'], on_epoch=True, prog_bar=True)  # 记录准确率
        self.log("f1", custom_metrics['f1'], on_epoch=True, prog_bar=True)  # 记录F1分数
        self.outputs.clear()  # 清空输出列表

    def predict(self, input_ids):
        pred = self.model.generate(input_ids, max_length=self.max_length)  # 生成预测结果
        decoded_output = self.tokenizer.batch_decode(pred, skip_special_tokens=True)  # 解码预测结果
        return decoded_output  # 返回解码后的输出

    def predict_step(self, batch, batch_idx):
        """
        预测步骤：
        - 对输入进行预测并返回解码后的文本
        """
        pred = self.model.generate(batch['input_ids'], max_length=self.max_length)  # 生成预测结果
        decoded_output = self.tokenizer.batch_decode(pred, skip_special_tokens=True)  # 解码预测结果
        return decoded_output  # 返回解码后的输出

    def configure_optimizers(self):
        """
        配置优化器：
        - 使用AdamW优化器
        """
        # 如果启用CPU卸载，可使用DeepSpeedCPUAdam（代码中未启用）
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)  # 使用AdamW优化器
        return optimizer  # 返回优化器
