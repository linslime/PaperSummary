## query优化
* 业务背景：工单查询。参考历史对话信息，优化 query
* 方法：构建 prompt

## 意图识别
* 业务背景：小工单查询智能体
* 方法：构建 prompt
* 模型：Qwen

## RAG
* 业务背景：中控智问项目。搭建企业知识库，进行知识问答
* 数据说明：
	* 产品说明书：控制系统、工业软件、仪器仪表等
	* FAQ数据集：SEP故障排查库、小8助手、IT运维、研发FAQ等
* 构建知识库
	* chunk：对于数据进行分块。
	* embeddings：
		* 利用 bge-m3 对知识进行向量化。
		* 或者 bce-embedding-base_v1
	* 数据库： 对数据进行向量化，最后存入数据库  elasticsearch
* 知识问答
	* query优化：
		* 文本格式优化：
			* 全角字母改为半角字母
			* 繁体改为简体
			* 检查字符串是否为“无效字符串”：
				* 纯数字字符串
				* 整串只包含标点符号或其他非字母、非数字、非空格字符
			* 同义词转换
				* 用 jieba 分词
				* 用同义词词表转换
		* 参考历史对话信息，构建 prompt 优化 query
	* retrivel：根据 query 在知识库中进行检索 faiss
	* 相似度匹配：cosin相似度
	* rerank：对检索出的数据进行重排序
		* WeightedRanker: 基于权重分配，通过计算加权平均值来合并不同向量场的搜索结果。
		* RRFRanker: 基于权重分配，通过计算加权平均值来合并不同向量场的搜索结果。
		* bce-reranker-base_v1：网易开源的算法，可以与bce-embedding配合
		* 利用最长公共子序列和最长功能子串进行加权，最后求和
	* LLM 生成：将检索出的数据和 query 送入 prompt 模板，再用大模型生成回复
* 效果：MRR（平均倒数排名）, 平均精确率（Context Precision），整体检索到的概率
## post-train
* 业务背景：中控智问智问回答。对于ChatGLM-6B模型基于 [P-Tuning v2](https://github.com/THUDM/P-tuning-v2)的微调，实现标准问答库的智能搜索功能。
* 数据：本次微调数据集来源于故障排查库标准问答，泛化后的训练集共8195条，其中训练集5214条、验证集1490条、测试集1491条。
	* supcon_faq 数据集任务为根据输入问题（content）给出标准答案（summary）。
```json
	{
    "content":"主控卡拷贝失败怎么办？",
    "summary":"1.工作卡故障。查看工作卡故障诊断是否正常，组态是否出错。2.备用主控卡故障。请联系研发人员协助分析原因。"}
```

* 方法：
	* 分布式训练：利用 swift 框架、结合 deepspeed 和 megatron 进行分布式回答
	* SFT 算法：使用 lora 算法进行微调
	* 偏好优化：使用 dpo 算法对模型进一步调整
	* 超参数调整：对超参数进行调整
	* 问题解决：出现灾难性遗忘
* 效果：使用 BLUE 和 ROUGH 对效果进行评测。
| 指标            | 测试结果  |
|---------------|-------|
| BLEU-4        | 87.12 |
| Rouge-1       | 90.80 |
| Rouge-2       | 88.04 |
| Rouge-l       | 89.71 |
## 数据生成
* 业务背景：CoT数据生成
* LLM 生成数据：
	* 原始数据收集：分为外部数据和内部数据。数据格式是 pdf 和 word 文档。利用 ocr 技术进行文本提取。
	* chunk：按照滑动窗口对文本进行分块。
	* 简单问题生成：对每一分块，构建 prompt 模板，利用大模型生成多个对于该分块的简单问题。
	* 复杂问题生成：利用大模型，根据简单问题，构建复杂问题。
	* think 过程生成。将复杂问题和对应的 chunk 构建 prompt，  利用大模型生成思考过程，并润色复杂问题表达。
* 利用开源数据集生成：
	* 原始数据收集：math 和 logic相关数据集。
	* 数据处理：构建 prompt，利用大模型生成数据。
	* 要求：
		* 要求大模型将中文翻译成英文
		* 要求将复杂问题拆解成简单问题
		* 要求给出思考过程
* 数据评价：构建 prompt ，利用 chatgpt-4o 对生成的数据进行评价
