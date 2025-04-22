## LLM 基础
### transformer 基础知识
1. LN 和 BN 区别？
2. RMS Norm 公式
3. 多头注意力机制公式和代码
4. 多购注意力机制为什么除以根号 d
5. qkv 矩阵的计算
6. ResNet 的作用？
7. RMS Norm 比 LayerNorm 好在哪里？
8. SwiGLU 激活函数公式？SwiGLU 比 ReLU 好在哪里？
9. RoPE 旋转式位置编码公式？好在哪里？
10. Causal Mask 机制介绍
11. 注意力机制的变体，比如MHA、MQA、GQA、MLA
12. Post-LN、Pre-LN区别
13. Causal-LM、Prefix-LM区别
14. Decoder-only，Encoder-only 和 Encoder-Decoder的区别是什么？为什么大部分是 Decoder-only?
15. 主流大模型的架构、训练任务、训练 trick
16. 为什么使用MHA?为什么用点乘？
17. 流行的模型对于 embedding 有什么改进？
18. 讲一讲几种位置编码
19. transformer 如何缓和过拟合、梯度爆炸、梯度消失？
20. 为什么 Attention 用点乘，而不是加法？
21. 介绍cross attention
22. BERT 原理与 NSP 和      MLM

### Post-Training
1. LoRA 初始化怎么做？用的lora_rank是多少？为什么不用其他的数？
2. 为什么用 LoRA？
3. LoRA、DPO 和 RL 在数学上的区别
4. 对 KL 散度的三种估计的理解
5. 对 PPO DPO GRPO 的理解
6. p-tuning v2 的初始化
7. GRPO 和 PPO 对比，为什么 critic model 可以去掉？为什么 PPO 要有critic model?critic model 和 reward model 相比有什么区别？为什么不去掉 reward model？
8. deepseek v1-v3 的改进？

### 分布式训练
1. transformer 如何做推理和训练加速？有什么区别？
2. deepspeed 的三个阶段
3. 介绍一下 deepspeed 和 megatron

### 模型代码题
1. MHA
2. LayerNorm
3. Transformer Encoder(MHA + LayerNorm + FFN)
4. PE
5. ROPE
6. SwiGLU
7. RMS Norm
8. cross attention