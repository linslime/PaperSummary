## query 改写
* **业务背景**：在多轮对话系统中，用户提问常常存在指代不清、上下文依赖强、信息不完整等问题，给下游意图识别和信息检索带来挑战。为提升大语言模型在复杂语境下的理解能力，我设计并实现了一套基于 Prompt 工程的 Query 改写模块，用于对用户提问进行上下文感知的优化重构。
* **方法**：我分几个步骤来设计这个 prompt
	* **角色设定**：我首先明确模型的角色是“语言处理模块”，并告知它的目标是对用户问题进行“优化改写”，让模型具备任务导向。
	* **输入结构说明**：我在 prompt 中引入两个关键输入：【用户问题】和【历史记录】，并明确说明历史记录的组成形式。
	* **相关性判断机制**：我设计了两种处理策略：当当前问题与历史记录有明显关联时，模型需要充分融合历史内容进行问题补全；如果几乎无关，就直接使用用户的原始提问。
	* **输出格式约束**：为了方便对接系统，我要求输出是结构化的 JSON 格式，统一格式为 {"Query": "优化后的问题"}。
	* **提供示例驱动模型行为**：我加入了两个典型示例，引导模型理解“模糊指代补全”和“无关问题直接输出”的行为模式，提高模型的一致性和泛化能力。
* **模型**：qwen2.5-72b-instruct
* **效果**：
	* 在内部测试集中，Query 改写模块显著提升系统对用户意图的识别准确率（+23%）；
	* 下游检索任务的召回率提升约 17%；

## 小工单查询智能体
* **业务背景**：本项目旨在构建一个基于自然语言交互的生产工单智能体，用于协助制造型企业员工快速完成生产工单创建、下发、查询等任务。通过对用户输入的理解和结构化处理，智能体可自动调度后端 API，实现从“语句到业务”的一站式闭环操作。其核心能力包括：
	* **意图识别**：理解用户的业务意图
	* **命名实体识别**：提取关键信息
	* **agent**：自动串联后端接口完成工单全流程处理
* **topic 分类**：
	* 构造 Prompt，描述所有支持的 Topic 及其典型输入，作为上下文；
	* 将用户输入拼接入 Prompt 中，调用大语言模型进行判断；
	* 大模型根据 Prompt 理解语义，返回意图类别；
	* 根据意图识别结果决定后续 action。
* **命名体识别**：
	* 基于大语言模型，通过构建 Prompt 实现对用户输入的结构化理解；
	* Prompt 中包括常见字段的描述、样例句子和提取要求；
	* 用户输入作为 Query 注入 Prompt 后，调用大模型输出结构化 JSON，包括字段名、值、类型等；
	* 模型自动判断字段类型（如产品名 vs 产品编码、日期 vs 时间范围等）；
	* 对于模糊输入、缺失字段，模型可输出部分信息并标记不确定项，为后续交互留口；
	* 支持嵌套结构处理，如参考 A 产品创建 B 产品工单的嵌套 NER。
* **模型**：qwen2.5-72b-instruct
* **效果**：在内部构造的真实业务测试集上，Topic 分类准确率达 94.6%，命名实体识别准确率达 92.3%；

## RAG
* **业务背景**：利用 RAG 技术实现的基于企业内部知识库的问答应用。
* **数据说明**：
	* **产品说明书**：控制系统、工业软件、仪器仪表等
	* **FAQ 数据集**：SEP 故障排查库、小8助手、IT运维、研发FAQ等
* **构建知识库**
	* **chunk**：对于数据进行分块。
	* **embeddings**：
		* 利用 bge-m3 对知识进行向量化。
		* 或者 bce-embedding-base_v1
	* **数据库**： 对数据进行向量化，最后存入数据库  elasticsearch
* **知识问答**
	* **query 处理**：
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
	* **retrivel**：根据 query 在知识库中进行检索 faiss
	* **相似度匹配**：cosin相似度
	* **rerank**：对检索出的数据进行重排序
		* **WeightedRanker**: 基于权重分配，通过计算加权平均值来合并不同向量场的搜索结果。
		* **RRFRanker**: 基于权重分配，通过计算加权平均值来合并不同向量场的搜索结果。
		* **bce-reranker-base_v1**：网易开源的算法，可以与bce-embedding配合
		* 利用最长公共子序列和最长功能子串进行加权，最后求和
	* **LLM 生成**：将检索出的数据和 query 送入 prompt 模板，再用大模型生成回复
* **模型**：qwen2.5-72b-instruct
* **效果**：MRR（平均倒数排名）, 平均精确率（Context Precision），整体检索到的概率
## post-train
* **业务背景**：中控智问智问回答。对于ChatGLM-6B模型基于 [P-Tuning v2](https://github.com/THUDM/P-tuning-v2) 的微调，实现标准问答库的智能搜索功能。
* **数据**：本次微调数据集来源于故障排查库标准问答，泛化后的训练集共8195条，其中训练集5214条、验证集1490条、测试集1491条。
	* supcon_faq 数据集任务为根据输入问题（content）给出标准答案（summary）。
	```json
		{
			"content":"主控卡拷贝失败怎么办？",
			"summary":"1.工作卡故障。查看工作卡故障诊断是否正常，组态是否出错。2.备用主控卡故障。请联系研发人员协助分析原因。"
		}
	```
* **方法**：
	* **分布式训练**：利用 swift 框架、结合 deepspeed 和 megatron 进行分布式回答
	* **SFT 算法**：使用 lora 算法进行微调
	* **偏好优化**：使用 dpo 算法对模型进一步调整
	* **超参数调整**：对超参数进行调整
	* **问题解决**：出现灾难性遗忘
* **实验参数设置**
* **P-Tuning v2 阶段**
	```
	pre_seq_len=128
	learning_rate=2e-2
	quantization_bit=4
	max_source_length=128
	max_target_length=256
	per_device_train_batch_size=16
	per_device_eval_batch_size=16
	num_train_epochs=10
	```
* **效果**：使用 BLUE 和 ROUGH 对效果进行评测。
| 指标           |测试结果|
|---------------|-------|
| BLEU-4        | 87.12 |
| Rouge-1       | 90.80 |
| Rouge-2       | 88.04 |
| Rouge-l       | 89.71 |
## 数据生成
* **业务背景**：CoT 数据生成
* **LLM 生成数据**：
	* **原始数据收集**：分为外部数据和内部数据。数据格式是 pdf 和 word 文档。利用 ocr 技术进行文本提取。
	* **简单问题生成**：使用 **deepseek R1** 对文章提出一系列简单问题
		* **chunk**：按照滑动窗口对文本进行分块。chunk_size = 3000，overlap = 500
		* **生成方法**： 对每一分块，构建 prompt 模板，利用 **deepseek R1** 生成多个对于该分块的简单问题。每 250 个字数要求生成 1 个 query。最后将结果保存为 json。
		* **举例说明**：
		```
		{
        	"question": "研究的时间范围是什么？",
        	"chunk": 0,
        	"hash": 2721229337,
        	"function": "document"
    	}
		```
	* **复杂问题生成**：利用 **deepseek R1**，根据简单问题，构建复杂问题。
		* **举例说明**：
		```
		{
        	"complex_question": "‘双碳’目标对企业市场营销有何影响？绿色营销模式的主要目标是什么？",
        	"source_questions": [
            {
                "question": "‘双碳’目标对企业市场营销有何影响？"
            },
            {
                "question": "绿色营销模式的主要目标是什么？"
            }
        	],
        	"chunk": 0,
        	"hash": 1073500502
    	}
		```
	* **think 过程生成**：将复杂问题和对应的 chunk 构建 prompt，  利用大模型生成思考过程，并润色复杂问题表达。
	* **举例说明**
		```
		{
        	"split_question": [
            	{
                	"question": "“双碳”目标对企业市场营销的具体影响有哪些？"
            	},
            	{
                	"question": "绿色营销模式的主要目标包括哪些内容？"
           		}
        	],
        	"complex_question": "‘双碳’目标对企业市场营销有何影响？绿色营销模式的主要目标是什么？",
        	"think": "好的，我需要处理用户提供的复杂问题：“‘双碳’目标对企业市场营销有何影响？绿色营销模式的主要目标是什么？”。首先，按照要求要将这个复合问题拆分成两个独立的简单问题。\n\n首先分析第一个部分：“‘双碳’目标对企业市场营销有何影响？”。这个问题询问的是双碳目标带来的具体影响，需要确保分解后的子问题明确且不含复合结构。可能的拆分是直接将其作为一个独立问题，因为它已经聚焦于影响方面，无法进一步分解。\n\n接下来处理第二部分：“绿色营销模式的主要目标是什么？”。同样，这个问题询问的是目标，应该单独作为一个问题，不需要拆分，因为它已经足够具体和独立。\n\n检查每个问题是否符合要求：简洁、独立、不含代词。确认后，这两个问题可以分别作为拆分后的结果。因此，最终的JSON结构应包含这两个问题。\n"
    	}
		```
* **利用开源数据集生成**：
	* **原始数据收集**：math 和 logic相关数据集。
	* **数据处理**：构建 prompt，利用大模型生成数据。
	* **要求**：
		* 要求大模型将中文翻译成英文
		* 要求将复杂问题拆解成简单问题
		* 要求给出思考过程
* **数据评价**：构建 prompt ，利用 chatgpt-4o 对生成的数据进行评价
## 模型
* 本地部署：
	* qwen2.5-72b-instruct
	* DeepSeek-R1-Distill-Qwen-32B：由 Qwen2.5-32B 蒸馏而来
* api调用：
	* deepseek r1
	* deepseek v3
	* chatgpt-4o

## hypergraph
* **业务背景** 先进行 query 重写， 构建 prompt 生成 sql 语句

## A-MEM: Agentic Memory for LLM Agents
* **论文：**A-MEM: Agentic Memory for LLM Agents
* **链接：**https://arxiv.org/abs/2502.12110v7
* **github：**https://github.com/agiresearch/A-mem.git
* **业务背景**：对 LLM 进行长对话记忆管理
* **新增记忆**：主要分为以下步骤
	1. **处理新记忆**：得到新的一轮对话，通过 LLM 分析 content 得到这段文本的 keywords、context、tags，得到新记忆。
	2. **检索旧记忆**：通过新记忆的 content，使用 Faiss 和 Elasticsearch 对现存记忆进行稠密检索和语义检索；
	3. **演化记忆**：由 LLM 判断新旧记忆的演化。
		* **新记忆演化**：为新记忆选择 link 记忆，更新 tags
		* **旧记忆演化**：更新 link 记忆的 context、tags
	4. **添加新记忆**：将新记忆的 content 向量化，创建索引，并存入 Faiss。
* **删除记忆**：通过 memory_id，从 Faiss 中删除对应的memory
* **更新记忆**：通过 memory id，先删除旧记忆，再新增记忆。
* **查询记忆**：通过 query 对各个记忆进行稀疏检索和稠密检索。
* **实验**：
	* **数据集**：LoCoMo dataset https://aclanthology.org/2024.acl-long.747/ DialSim dataset
	* **模型**：
		* Llama 3.2 3b
		* Llama 3.2 1b
		* Qwen2.5 3b
		* Qwen2.5 1.5b
		* GPT 4o
		* GPT-4o-mini
	* **baseline**
		* LOCOMO
		* READAGENT
		* MEMORYBANK
		* MEMGPT
	* **指标**：
		* F1
		* BLEU
		* ROUGE
		* METEOR
		* SBERT Similarity
	* **对比角度**：
		* Performance Analysis
		* Cost-Efficiency Analysis：分析时间复杂度
	* **消融实验**：评估 Link Generation (LG) and Memory Evolution (ME) 两个模块的有效性
	* **超参数分析**：探索检索记忆时，top k对效果的影响。虽然增加 k 值通常会提升性能，但这种提升会逐渐趋于平稳，有时甚至会在更高 k 值时略有下降
	* 分析空间复杂度
	* 记忆可视化
## function call

## MiniMind
* **github：** https://github.com/jingyaogong/minimind.git
* **Pretrain：**
	| 超参数 | 值 |
	| :- | :- |
	| 优化器 | AdamW |
	| epochs | 2-6 |
	| batch_size | 32 |
	| learning_rate |5e-4 |
	| accumulation_steps | 8 |
	|loss | CrossEntropyLoss |
	
* **SFT：**
	| 超参数 | 值 |
	| :- | :- |
	| 优化器 | AdamW |
	| epochs | 2 |
	| batch_size | 16 |LoRA
	| learning_rate |5e-6 -> 5e-7 |
	| accumulation_steps | 1 |
	|loss | CrossEntropyLoss |
	
* **LoRA：**
	| 超参数 | 值 |
	| :- | :- |
	| optimizer | AdamW |
	| epochs | 10 |
	| batch_size | 32 |
	| learning_rate |1e-4 |
	| accumulation_steps | 1 |
	|loss | CrossEntropyLoss |
	| rank | 8 |
	
* **DPO：**
	| 超参数 | 值 |
	| :- | :- |
	| optimizer | AdamW |
	| epochs | 2 |
	| batch_size | 4 |
	| learning_rate |1e-8 |
	| accumulation_steps | 1 |
	|loss | CrossEntropyLoss |

* **DPO：**
	| 超参数 | 值 |
	| :- | :- |
	| optimizer | AdamW |
	| epochs | 2 |
	| batch_size | 4 |
	| learning_rate |1e-8 |
	| accumulation_steps | 1 |
	
## Memories for Virtual AI Characters
* **论文：**Memories for Virtual AI Characters
* **链接：**https://aclanthology.org/2023.inlg-main.17/
* **问题：**提出了一个增强虚拟AI角色长期记忆的系统，使其能够记住关于自身、世界和过往经历的事实。
* **方法：**
	* **主要思路：**通过向通用的 LLM 发出基于聊天上下文和相关记忆动态创建的提示，来生成角色响应。
	* **Chat History：**聊天记录包含当前对话中所有仍需转换为记忆的消息。这些聊天记录在整个系统中用作短期记忆，并作为用户最后一条消息的直接上下文。
	* **Query Creator：**一旦聊天记录更新了最新的用户消息，查询创建器就会生成一个搜索查询，用于检索相关的记忆。
	* **Retrieving Memories：**用 embeddings 模型将 query 和 memories 向量化，将memories 分为多个知识源，对每个知识源里的知识计算相似度，检索出多条记忆
	*  **Re-ranking Memories：**对记忆进行重排序
	*  **Character Response Generation：**构建提示词
	*  **Memory Creation Pipeline：**对记忆进行分块，然后总结出一个个的事实，并保证每一个都是可以被理解的
	*  **Memory Structure：**每个记忆包含fact、embedding 和 meta information
	*  **Forgetting Model：**对长时间不访问的记忆进行遗忘，指数遗忘
* **实验：**
	* **评估角度**：
		* query 和 检索到的记忆的契合度
		* LLM 在准确引用所用记忆方面的有效性
	* **评估方法**：
		1. 首先，从角色的回应中提取未经验证的主张。用GPT-4讲回复分解为独立的句子。
		2. 用 GPT-4 评价句子的好坏
		3. 人工核验
	* **评估结果**：从检索的契合度和引用的准确性分析

## 天工 Agent 逆向
* **官网：**https://www.tiangong.cn/
* **主题理解与信息收集初始化：**
	* **目标：**从用户输入的简要主题出发，主动提问补全研究所需关键信息
	* **实现：**提供一个表单，让用户补充相关信息，如文章题目、写作语言、写作篇幅、分析重点、分析角度、写作风格、其他要求
	* **思考：**补充的信息用于构建 Prompt，或者用于选择模型
* **研究规划：**
	* **目标：**根据补全后的参数制定研究路径和写作架构，并获得用户确认
	* **实现：**拆分问题为多个子任务，可递归拆分，形成研究和写作计划 
	* **思考：**子任务拆分的高效实现方式未知
* **执行计划：**
	* **目标：**结合多个工具，完成”搜索 --> 浏览 --> 分析 -->反馈“循环执行
	* **主要功能模块：**
		* **WebSearch：**网络搜索
		* **WebBrowser：**网页内容阅读
		* **信息聚合：**将多个来源的信息进行聚合
		* **回溯：**如果发现之前的聚合的信息有问题，那么重新搜索相关网页，重新整理信息
	* **实现：**集成相关 MCP Server，实现网络搜索，网页内容阅读等功能；利用大模型完成信息分析，对缺失或者偏差内容进行回溯
	* **思考：**需要调研可靠的 web search和 web browser MCP Server；如何实现对网页内容的理解和聚合；如何实现对历史任务的低质量内容进行回溯重处理；在执行计划时，并非线性的执行，也会有一个前后文的关联，如何实现？
* **最终报告的生成：**
	* **目标：**基于子任务的结果，生成最终报告
	* **实现：**未知
	* **思考：**不知道怎么实现最终报告的生成；如果子任务结果篇幅短，可不可以将全文一起送入 LLM?如果子任务结果的篇幅很长，又要以什么策略调度 LLM 呢？
* **最终思考：**整理流程有一个了解，但是对于细节缺少认识；之后调研一下开源的论文或者 github，熟悉相关技术

## Think-in-Memory: Recalling and Post-thinking Enable LLMs with Long-Term Memory
* **论文：**Think-in-Memory: Recalling and Post-thinking Enable LLMs with Long-Term Memory
* **链接：**https://arxiv.org/abs/2311.08719
* **问题：**提出一个长对话记忆管理系统，不直接存储对话，要经过思考，也就是经过 LLM 处理成拥有完整的信息；
* **方法：**
	* **回忆和生成：**从记忆中检索相关想法，送入大模型生成答案
	* **思考和更新：**模型生成回复后，调用 LLM 进行思考，插入、移除、融合记忆。
	* **PEFT：**对模型进行高效参数微调
* **关键组成部分：**
	* **”思考“：**将原本的记忆重新梳理成原子知识
	* **检索方式：**LSH

## THEANINE: Revisiting Memory Management in Long-term Conversations with Timeline-augmented Response Generation
* **论文：**THEANINE: Revisiting Memory Management in Long-term Conversations with Timeline-augmented Response Generation
* **链接：**https://arxiv.org/abs/2406.10996v1
* **问题：**通过构建相关历史事件和因果关系的记忆时间线，增强 LLM 响应生成。
* **方法：**
	* **构建记忆网络：**
		* 识别联想记忆，为记忆构建连接
		* 关系感知记忆连接
	* **时间线检索和时间线优化：**
		* 检索原始记忆时间线。
		* 上下文感知的时间线细化。
		* 时间线增强响应生成

## Keep Me Updated! Memory Management in Long-term Conversations
* **论文：**Keep Me Updated! Memory Management in Long-term Conversations
* **链接：**https://aclanthology.org/2022.findings-emnlp.276/
* **问题：**将记忆表示为关键信息的非结构化文本描述，选择性的消除无效或者冗余信息
* **方法：**
	* 生成回复，
	* 对回复进行摘要
	* 更新记忆库
	* 检索记忆库，生成回复