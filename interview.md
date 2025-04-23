## LLM 基础
### transformer 基础知识
1. LN 和 BN 区别？
3. 多头注意力机制公式和代码
4. 多购注意力机制为什么除以根号 d
5. qkv 矩阵的计算
6. ResNet 的作用？
7. RMS Norm 比 LayerNorm 好在哪里？
8. SwiGLU 比 ReLU 好在哪里？
9. RoPE 旋转式位置好在哪里？
10. Causal Mask 机制介绍
11. 注意力机制的变体，比如MHA、MQA、GQA、MLA
12. Post-LN、Pre-LN区别
13. Causal-LM、Prefix-LM区别
14. Decoder-only，Encoder-only 和 Encoder-Decoder的区别是什么？为什么大部分是 Decoder-only?
15. 主流大模型的架构、训练任务、训练 trick
16. 为什么使用MHA?为什么用点乘？
17. 流行的模型对于 embedding 有什么改进？
18. 讲一讲几种位置编码 PoPE、PE、Alibi
19. transformer 如何缓和过拟合、梯度爆炸、梯度消失？
20. 为什么 Attention 用点乘，而不是加法？
21. 介绍cross attention
22. BERT 原理与 NSP 和  MLM
23. BERT 和 GPT 的区别，可以从训练目标回答
24. 介绍 LLama2 架构
25. BERT 模型中文本到 id 的转换是怎么样的？
26. 大模型的词表有哪些？有什么不同？
27. BERT 和 Ernie 有什么不同？
28. LSTM 的优势和不足？
29. 介绍一下 Scaling Laws
30. 为什么 BERT 的三个 Embedding 可以相加？
31. 为什么需要 RLHF？
32. 为什么 DPO 比较稳定呢？从算法原理，参数设置的角度回答
33. 介绍一下 Top-K、Top-P、Temperature
34. softmax 公式
35. ReLU为什么可以缓解梯度消失问题？
36. 交叉熵数学公式

### Training
1. LoRA 初始化怎么做？用的lora_rank是多少？为什么不用其他的数？
2. 为什么用 LoRA？
3. LoRA、DPO 和 RL 在数学上的区别
4. 对 KL 散度的三种估计的理解
5. 对 PPO DPO GRPO 的理解
6. p-tuning v2 的初始化
7. GRPO 和 PPO 对比，为什么 critic model 可以去掉？为什么 PPO 要有critic model？critic model 和 reward model 相比有什么区别？为什么不去掉 reward model？
8. deepseek v1-v3 的改进？
9. 如何估算 7B 模型显存需要
10. 介绍一下 RLHF
11. 介绍主流大模型的 loss
12. 介绍半精度训练
13. transformer 如何做推理和训练加速？有什么区别？
14. deepspeed 的三个阶段
15. 介绍一下 deepspeed 和 megatron
16. LLM 的训练目标是什么？
17. 涌现能力是为什么？灾难性遗忘是为什么？
18. fp16 和 bf16 的区别？
19. RAG 和 SFT 有什么优势？
20. 什么是大模型幻觉问题？如何解决？
21. 什么是模型复读问题？如何解决？
22. Pretrain 后为什么要 SFT？
23. 为什么用 DPO 而不用 PPO？出发点是什么？
24. 模型训练时长？遇到的问题，以及解决方案？
25. LoRA 的超参数设置，为什么是 8 ，有没有选过其他参数？
26. 预估模型训练显存大小
27. 损失函数和学习率扩大十倍，影响一样吗？
28. 介绍一下 Adam 优化器
29. 预训练的步骤
30. 介绍一下 o1
31. 介绍一下 self-play 自我博弈
32. 大模型数据合成有哪些方法
33. SFT 的 loss 如何设置？
34. 如何优化大模型训练速度

### 公式
1. DPO 公式
2. softmax 公式
3. 多头注意力机制公式
4. SwiGLU 激活函数公式
5. RMS Norm 公式
6. Layer Norm 公式
7. Batch Norm 公式
8. RoPE 旋转式位置编码公式
9. 交叉熵数学公式
10. BLEU 和 ROUGH 的计算公式
11. PPO 公式

### 模型评测
1. 如何看大模型好坏？评估指标有哪些
2. BLEU 和 ROUGH 的计算公式
3. 数据集如何构建、评测？数据集评测过程中遇到哪些困难？如何解决？
4. RAG 如何评价

### 模型代码题
1. MHA
2. LayerNorm
3. Transformer Encoder(MHA + LayerNorm + FFN)
4. PE
5. ROPE
6. SwiGLU
7. RMS Norm
8. cross attention
9. Tokenizer
10. data_loader
11. trainer
12. 交叉熵代码