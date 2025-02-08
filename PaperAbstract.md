# PaperSummary
___
## RAG-RLRC-LaySum at BioLaySumm: Integrating Retrieval-Augmented Generation and Readability Control for Layman Summarization of Biomedical Texts
原文：https://aclanthology.org/2024.bionlp-1.75/
### abstract
本文介绍了 RAG-RLRC-LaySum 框架，旨在通过先进的自然语言处理 (NLP) 技术让普通人也能理解复杂的生物医学研究。我们创新的检索增强生成 (RAG) 解决方案通过重新排序方法增强，利用多个知识源来确保普通摘要的准确性和相关性。此外，我们的强化学习可读性控制 (RLRC) 策略提高了可读性，使非专业人士也能理解科学内容。使用可公开访问的 PLOS 和 eLife 数据集进行的评估表明，我们的方法超越了 Plain Gemini 模型，可读性分数提高了 20%，ROUGE-2 相关性分数提高了 15%，事实准确性提高了 10%。RAG-RLRC-LaySum 框架有效地使科学知识民主化，增强了公众对生物医学发现的参与度。
___
## ClinicalRAG: Enhancing Clinical Decision Support through Heterogeneous Knowledge Retrieval
原文：https://aclanthology.org/2024.knowllm-1.6/
### abstract
大型语言模型 (LLM) 已经彻底改变了不同领域的文本生成，展示了以惊人准确性模仿人类文本的能力。然而，这些模型经常遇到一个重大障碍：产生幻觉，这一缺陷在精确度至关重要的医疗保健领域尤其有害。在本文中，我们介绍了 ClinicalRAG，这是一种新颖的多智能体管道，通过将异构医学知识（结构化和非结构化）整合到 LLM 中来解决这个问题，以提高诊断准确性。ClinicalRAG 可以从用户输入中提取相关医学实体，并在文本生成过程中动态集成相关医学知识。比较分析表明，ClinicalRAG 的表现明显优于知识缺乏的方法，在临床决策支持方面提供了更高的可靠性。这一进步标志着朝着减轻 LLM 医疗应用中的错误信息风险迈出了关键的概念验证步骤。
___
## The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation (RAG)
原文：https://aclanthology.org/2024.findings-acl.267/
### abstract
检索增强生成 (RAG) 是一种强大的技术，可用于利用专有和私有数据促进语言模型生成，其中数据隐私是一个关键问题。尽管大量研究已经证明了大型语言模型 (LLM) 存在隐私风险，但 RAG 技术可能会重塑 LLM 生成的固有行为，带来目前尚未得到充分探索的新隐私问题。为此，我们使用新颖的攻击方法进行了广泛的实证研究，证明了 RAG 系统在泄露私有检索数据库方面的脆弱性。尽管 RAG 给检索数据带来了新的风险，但我们进一步发现，RAG 可用于减轻旧风险，即 LLM 训练数据的泄露。总的来说，我们在本文中揭示了许多关于检索增强 LLM 隐私保护的新见解，这可能使 LLM 和 RAG 系统构建者都受益。
___
## DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models
原文：https://aclanthology.org/2024.acl-long.702/
### abstract
动态检索增强生成 (RAG) 范式在大型语言模型 (LLM) 的文本生成过程中主动决定何时检索以及检索什么。此范式有两个关键要素：确定激活检索模块的最佳时机（决定何时检索）和在触发检索后设计适当的查询（确定检索什么）。然而，当前的动态 RAG 方法在这两方面都存在不足。首先，决定何时检索的策略通常依赖于静态规则。此外，决定检索什么的策略通常局限于 LLM 的最近一句话或最后几个标记，而 LLM 的信息需求可能跨越整个上下文。为了克服这些限制，我们引入了一个新框架 DRAGIN，即基于 LLM 信息需求的动态检索增强生成。我们的框架专门设计用于根据文本生成过程中 LLM 的信息需求来决定何时检索什么。我们在 4 个知识密集型生成数据集上对 DRAGIN 和现有方法进行了全面评估。实验结果表明，DRAGIN 在所有任务上都取得了优异的表现，证明了我们方法的有效性。
___
## RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models
原文：https://aclanthology.org/2024.acl-long.585/
### abstract
检索增强生成 (RAG) 已成为缓解大型语言模型 (LLM) 中幻觉的主要技术。尽管集成了 RAG，LLM 仍可能对检索到的内容提出不受支持或相互矛盾的主张。为了在 RAG 下制定有效的幻觉预防策略，创建可以衡量幻觉程度的基准数据集非常重要。本文介绍了 RAGTruth，这是一个专门为分析 LLM 应用的标准 RAG 框架内各个领域和任务中的单词级幻觉而定制的语料库。RAGTruth 包含使用 RAG 从各种 LLM 自然生成的近 18,000 个响应。这些响应在个案和单词级别都经过了细致的手动注释，并结合了对幻觉强度的评估。我们不仅对不同 LLM 中的幻觉频率进行了基准测试，还批判性地评估了几种现有幻觉检测方法的有效性。我们表明，使用 RAGTruth 等高质量数据集，可以对相对较小的 LLM 进行微调，并与使用 GPT-4 等最先进 LLM 的现有基于提示的方法相比，实现具有竞争力的幻觉检测性能。此外，经过微调的模型可以有效缓解 LLM 响应中的幻觉。
___
## M-RAG: Reinforcing Large Language Model Performance through Retrieval-Augmented Generation with Multiple Partitions
原文：https://aclanthology.org/2024.acl-long.108/
### abstract
检索增强生成 (RAG) 通过从外部数据库检索相关记忆来增强大型语言模型 (LLM)。然而，现有的 RAG 方法通常将所有记忆组织在整个数据库中，这可能会限制对关键记忆的关注并引入噪音。在本文中，我们为 RAG 引入了一种多分区范式 (称为 M-RAG)，其中每个数据库分区都是 RAG 执行的基本单元。基于此范式，我们提出了一个新颖的框架，该框架利用具有多智能体强化学习的 LLM 来明确优化不同的语言生成任务。通过对七个数据集进行的全面实验，涵盖三个语言生成任务并涉及三个不同的语言模型架构，我们确认 M-RAG 始终优于各种基线方法，分别在文本摘要、机器翻译和对话生成方面实现了 11%、8% 和 12% 的改进。
___
## MobileSpeech: A Fast and High-Fidelity Framework for Mobile Zero-Shot Text-to-Speech
原文：https://aclanthology.org/2024.acl-long.733/
### abstract
零样本文本转语音 (TTS) 因其强大的语音克隆功能而备受关注，只需要几秒钟的看不见的说话人语音提示。然而，之前的所有工作都是为基于云的系统开发的。以自回归模型为例，虽然这些方法实现了高保真语音克隆，但它们在推理速度、模型大小和鲁棒性方面存在不足。因此，我们首次提出了 MobileSpeech，这是一种基于移动设备的快速、轻量级且鲁棒的零样本文本转语音系统。具体来说：1) 利用离散编解码器，我们设计了一个称为 SMD 的并行语音掩码解码器模块，它在生成过程中结合了来自语音编解码器的分层信息和跨不同编解码器层的权重机制。此外，为了弥合文本和语音之间的差距，我们引入了一个高级概率掩码，模拟语音生成过程中信息流从少到多的进展。 2) 对于说话者提示，我们从提示语音中提取细粒度的提示持续时间，并通过 SMD 中的交叉注意将文本、提示语音合并在一起。我们在多语言数据集上展示了 MobileSpeech 在不同级别的有效性，在生成速度和语音质量方面取得了最先进的成果。MobileSpeech 在单个 A100 GPU 上实现了 0.09 的 RTF，并且我们已成功在移动设备上部署了 MobileSpeech。音频样本可在 https://mobilespeech.github.io/ 上找到。
___
## SpikeVoice: High-Quality Text-to-Speech Via Efficient Spiking Neural Network
原文：https://aclanthology.org/2024.acl-long.429/
### abstract
受大脑启发的脉冲神经网络 (SNN) 已在视觉、自然语言和语音理解任务中证明了其有效性和效率，表明它们具有“看”、“听”和“读”的能力。在本文中，我们设计了 SpikeVoice，它通过 SNN 执行高质量的文本转语音 (TTS)，以探索 SNN“说话”的潜力。使用 SNN 进行此类生成任务的主要障碍在于需要模型掌握长期依赖关系。然而，脉冲神经元的串行性质导致未来脉冲时间步骤中的信息不可见，从而限制 SNN 模型只能在同一时间步骤内捕获序列依赖关系。我们将这种现象称为“部分时间依赖性”。为了解决这个问题，我们在 SpikeVoice 中引入了脉冲时间序列注意力 (STSA)。据我们所知，SpikeVoice 是 SNN 领域的第一部 TTS 作品。我们使用四个成熟的数据集进行实验，这些数据集涵盖中文和英文，涵盖单说话人和多说话人配置的场景。结果表明，SpikeVoice 可以实现与人工神经网络 (ANN) 相当的结果，而能耗仅为 ANN 的 10.5%。我们的演示和代码均可作为补充材料提供。
___
## Uni-Dubbing: Zero-Shot Speech Synthesis from Visual Articulation
原文：https://aclanthology.org/2024.acl-long.543/
### abstract
在语音合成领域，人们越来越重视使用多模态语音来增强鲁棒性。该领域的一个关键挑战是将音频与相应视频配对的数据集稀缺。我们采用了一种在多模态数据集的预训练阶段结合模态对齐的方法，通过在预训练权重内冻结视频模态特征提取组件和编码器模块的过程，以独特的方式促进零样本泛化，从而实现有效的跨模态和跨语言迁移。我们将这种方法命名为“Uni-Dubbing”。我们的方法可以对多模态和单模态音频数据进行微调。在多模态场景中，它的字错误率 (WER) 降低到 31.73%，超过了之前最好的 33.9%。它在音质和同步等指标方面也表现出色。在单模态音频中，它的 WER 达到了 36.08%，展示了对有限数据的适应性。它的领域泛化能力在视频翻译和音频生成中的各种语言任务中得到了证实。经过 433 小时音频数据的训练，它超越了使用 200 小时视听数据的技术。代码和演示可在 https://diracer.github.io/unidubbing 上找到。
___
## VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild
原文：https://aclanthology.org/2024.acl-long.673/
### abstract
我们引入了 VoiceCraft，这是一种标记填充神经编解码器语言模型，在有声读物、网络视频和播客的语音编辑和零样本文本转语音 (TTS) 方面均实现了最佳性能。VoiceCraft 采用 Transformer 解码器架构，并引入了一种标记重排程序，该程序结合了因果掩蔽和延迟堆叠，可在现有序列内生成。在语音编辑任务中，VoiceCraft 生成的编辑语音在自然度方面几乎与未编辑的录音没有区别，这由人类评估；对于零样本 TTS，我们的模型优于之前的 SotA 模型，包括 VALL-E 和流行的商业模型 XTTS v2。至关重要的是，这些模型是在具有挑战性和现实的数据集上进行评估的，这些数据集包含不同的口音、说话风格、录音条件以及背景噪音和音乐，与其他模型和真实录音相比，我们的模型表现始终良好。特别是，对于语音编辑评估，我们引入了一个高质量、具有挑战性且现实的数据集，名为。我们鼓励读者在 https://jasonppy.github.io/VoiceCraft_web 上收听演示。数据、代码和模型权重可在 https://github.com/jasonppy/VoiceCraft 上找到。
___
## emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation
原文：https://aclanthology.org/2024.findings-acl.931/
### abstract
我们提出了一个通用的语音情感表示模型emotion2vec。emotion2vec通过自监督的在线蒸馏在开源无标注情感数据上进行预训练，预训练过程中结合了话语级损失和帧级损失。emotion2vec在主流IEMOCAP数据集上仅对语音情感识别任务训练线性层，便超越了目前最先进的预训练通用模型和情感专家模型。此外，emotion2vec在10种不同语言的语音情感识别数据集中表现出一致的提升。emotion2vec在其他情感任务上也表现出优异的效果，比如歌曲情感识别、对话中的情感预测、情感分析等。对比实验、消融实验和可视化全面展示了所提出的emotion2vec的通用能力。据我们所知，emotion2vec是第一个在各类情感相关任务中通用的表示模型，填补了该领域的空白。
___
## TransFace: Unit-Based Audio-Visual Speech Synthesizer for Talking Head Translation
原文：https://aclanthology.org/2024.findings-acl.593/
### abstract
直接语音到语音翻译通过引入从自监督学习中获得的离散单元实现了高质量的结果。然而，与音频语音相比，将视听语音（即头部特写视频）从一种语言转换为另一种语言的头部特写翻译仍然面临着几个挑战：（1）现有方法总是依赖于级联，通过音频和文本进行合成，导致延迟和级联错误。（2）头部特写翻译的参考帧集有限。如果生成的翻译超出了原始语音的长度，则需要通过重复帧来补充视频序列，从而导致视频转换不协调。在这项工作中，我们提出了一个头部特写翻译模型 TransFace，它可以将视听语音直接翻译成其他语言的视听语音。它由一个语音到单元翻译模型和一个基于单元的视听语音合成器 Unit2Lip 组成，前者用于将音频语音转换为离散单元，后者用于并行地从离散单元重新合成同步的视听语音。此外，我们引入了有界持续时间预测器，确保等距说话头部翻译并防止重复参考帧。实验表明，Unit2Lip 显著提高了同步性，并在 LRS2 上将推理速度提高了 4.35 倍。此外，TransFace 在 LRS3-T 上为 Es-En 和 Fr-En 实现了令人印象深刻的 BLEU 分数 61.93 和 47.55，并实现了 100% 等时翻译。示例可在 https://transface-demo.github.io 上获取。
___
## The MERSA Dataset and a Transformer-Based Approach for Speech Emotion Recognition
原文：https://aclanthology.org/2024.acl-long.752/
### abstract
语音情感识别 (SER) 领域的研究依赖于全面的数据集，以便设计准确的情感检测模型。本研究介绍了多模态情感识别和情绪分析 (MERSA) 数据集，其中包括两周内收集的 150 名参与者的自然和脚本语音记录、转录文本、生理数据和自我报告的情感调查。这项工作还提出了一种新颖的情感识别方法，该方法使用基于 Transformer 的模型，集成预训练的 wav2vec 2.0 和 BERT 进行特征提取，并附加 LSTM 层以从语音和文本的融合表示中学习隐藏表示。我们的模型从唤醒、效价和支配维度预测情绪。我们在 MSP-PODCAST 数据集上训练和评估了该模型，并在一致性相关系数 (CCC) 方面取得了与表现最佳的模型相当的结果。此外，本文通过对 IEMOCAP 和 MERSA 数据集的跨域评估证明了该模型的有效性。
___
## HGP-NLP at BioLaySumm: Leveraging LoRA for Lay Summarization of Biomedical Research Articles using Seq2Seq Transformers
原文：https://aclanthology.org/2024.bionlp-1.78/
### abstract
通俗摘要旨在为非专家生成技术文章摘要，使普通受众能够轻松理解。研究中使用的技术语言通常会阻碍科学知识的有效交流，使非专家难以理解。自动通俗摘要可以增强对科学文献的访问，促进跨学科知识共享和公众理解。鉴于当前全球对清晰的医疗信息的需求，这对于生物医学文章来说尤为重要。大型语言模型 (LLM) 具有出色的语言理解能力，非常适合抽象摘要，有助于让公众获取复杂的信息。本文详细介绍了我们对 BioLaySumm 2024 共享任务：生物医学研究文章的通俗摘要的提交。我们在各种训练数据集设置和优化方法（如 LoRA）中对 T5 等序列到序列模型进行了微调和评估，以实现通俗摘要。我们的提交总体排名第 53 位。
___
## Saama Technologies at BioLaySumm: Abstract based fine-tuned models with LoRA
原文：https://aclanthology.org/2024.bionlp-1.72/
### abstract
尽管这些研究文章可能对公众有益，但由于生物医学研究文章使用技术术语和背景知识要求，因此对其进行通俗摘要是一项具有挑战性的问题。我们在参加 BioLaySumm 2024 时致力于解决这个问题。我们尝试了各种微调方法来为生物医学研究文章生成更好的通俗摘要。经过几次实验，我们根据给定文章的摘要构建了一个具有无监督微调的 LoRA 模型，然后使用后处理单元删除重复的句子。我们的模型在 BioLaySumm 2024 排行榜上排名第三。我们分析了我们尝试过的不同方法，并提出了一些进一步改进模型的想法。
___
## RA-LoRA: Rank-Adaptive Parameter-Efficient Fine-Tuning for Accurate 2-bit Quantized Large Language Models
原文：https://aclanthology.org/2024.findings-acl.933/
### abstract
部署具有大量参数和高内存需求的大型语言模型 (LLM) 会对计算效率提出挑战，尤其是在资源有限的特定应用的微调中。低秩自适应 (LoRA) 等技术通过训练基础模型的较小、可修改的扩展来减少内存使用量。但是，将量化与 LoRA 相结合，尤其是在低位场景中，可能会因量化误差而导致性能损失。我们创新的秩自适应 LoRA (RA-LoRA) 通过使用秩子空间分析动态调整适配器的秩来解决此问题，从而以更少的参数优化性能。我们在最先进的 LLM 上测试了 RA-LoRA 以进行 2 位高效微调，结果表明它可以以最少的可训练参数提高模型准确性，标志着量化感知微调方法的飞跃，并凸显了秩动态在优化量化 LLM 中的重要性。
___
## ResLoRA: Identity Residual Mapping in Low-Rank Adaption
原文：https://aclanthology.org/2024.findings-acl.525/
### abstract
低秩自适应 (LoRA) 是最流行的参数高效微调 (PEFT) 方法之一，常用于微调大型语言模型 (LLM)。然而，由于原始模型的计算路径较长，有效且快速地更新 LoRA 块的权重具有挑战性。为了解决这个问题，我们提出了 ResLoRA，一种改进的 LoRA 框架。通过在训练期间添加残差路径，并在推理过程中使用合并方法消除这些额外路径，与 LoRA 相比，我们的方法可以在更少的训练步骤中取得更好的结果，而无需任何额外的可训练参数或推理成本。在 NLG、NLU 和文本到图像任务上的实验证明了我们方法的有效性。据我们所知，ResLoRA 是第一个将残差路径与 LoRA 相结合的作品。我们方法的代码可在 [此网址](https://github.com/microsoft/LMOps/tree/main/reslora) 获得。
___
## LoraRetriever: Input-Aware LoRA Retrieval and Composition for Mixed Tasks in the Wild
原文：https://aclanthology.org/2024.findings-acl.263/
### abstract
低秩自适应 (LoRA) 为微调大型语言模型 (LLM) 提供了一种有效而高效的解决方案。LoRA 的模块化和即插即用特性使得能够集成各种特定领域的 LoRA，以增强 LLM 的功能。之前关于利用多个 LoRA 的研究要么侧重于特定的孤立下游任务，要么在训练期间固定 LoRA 的选择。然而，在现实世界中，LLM 会收到涵盖不同任务的不同提示，并且候选 LoRA 池通常会动态更新。为了弥补这一差距，我们提出了 LoraRetriever，这是一个检索然后组合的框架，可根据输入提示自适应地检索和组合多个 LoRA。LoraRetriever 包含三个主要组件：首先，识别和检索与给定输入相关的 LoRA；其次，制定有效整合检索到的 LoRA 的策略；第三，开发高效的批量推理以适应异构请求。实验结果表明，LoraRetriever 的表现始终优于基线，凸显了其实用性和多功能性。我们的代码可在 https://github.com/StyxXuan/LoraRetriever 上找到。
___
## STAR: Constraint LoRA with Dynamic Active Learning for Data-Efficient Fine-Tuning of Large Language Models
原文：https://aclanthology.org/2024.findings-acl.209/
### abstract
尽管大型语言模型 (LLM) 已经通过提示方法展示了小样本学习的强大能力，但对于复杂的推理任务，监督训练仍然是必要的。由于其广泛的参数和内存消耗，参数高效微调 (PEFT) 方法和内存高效微调方法都已被提出用于 LLM。然而，数据高效微调的目标——大量注释数据消耗的问题仍未得到探索。一种显而易见的方法是将 PEFT 方法与主动学习相结合。然而，实验结果表明，这种组合并不简单，而且会产生较差的结果。通过探测实验，这种观察结果可以用两个主要原因来解释：不确定性差距和模型校准不佳。因此，在本文中，我们提出了一种有效整合基于不确定性的主动学习和 LoRA 的新方法。具体而言，对于不确定性差距，我们引入了一种动态不确定性测量，它在主动学习迭代过程中结合了基础模型的不确定性和完整模型的不确定性。对于模型校准较差的问题，我们在 LoRA 训练过程中加入了正则化方法，以防止模型过于自信，并采用蒙特卡洛 dropout 机制来增强不确定性估计。实验结果表明，所提出的方法在三个复杂推理任务上的表现优于现有的基线模型。
___
## LoRAPrune: Structured Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning
原文：https://aclanthology.org/2024.findings-acl.178/
### abstract
大型语言模型 (LLM)，例如 LLaMA 和 T5，通过微调在各种任务中表现出色。尽管低秩自适应 (LoRA) 已经出现，可以在下游任务上以低成本对这些 LLM 进行微调，但它们的部署仍然受到巨大模型规模和计算成本的阻碍。训练后模型修剪提供了一种压缩 LLM 的方法。但是，当前为 LLM 设计的修剪方法与 LoRA 不兼容。这是因为它们在 LLM 上使用非结构化修剪，阻碍了 LoRA 权重的合并，或者它们依赖预训练权重的梯度来指导修剪，这可能会带来很大的内存开销。为此，我们提出了 LoRAPrune，这是一个新框架，可以以高度内存高效的方式提供准确的结构化修剪模型。具体来说，我们首先设计了一个LoRA引导的剪枝标准，该标准使用LoRA的权重和梯度而不是预训练权重的梯度来估计重要性。随后，我们将此标准集成到迭代剪枝过程中，有效地去除了冗余通道和头。大量实验结果表明，我们的LoRAPrune在LLaMA系列模型上的性能优于现有方法。在50％的压缩率下，LoRAPrune表现出优于LLM-Pruner的性能，在WikiText2上将困惑度降低了4.81，在PTB上将困惑度降低了3.46，同时还降低了52.6％的内存使用量。此外，LoRAPrune还匹配跨多个LLM的半结构化剪枝，证明了其广泛的适用性。代码可在https://github.com/aim-uofa/LoRAPrune获得。
___
## LoRA Meets Dropout under a Unified Framework
原文：https://aclanthology.org/2024.findings-acl.119/
### abstract
凭借卓越的能力，大型语言模型 (LLM) 已成为众多 NLP 应用中必不可少的元素，而参数高效的微调，尤其是 LoRA，作为一种轻量级的模型定制方法而广受欢迎。同时，各种 dropout 方法最初设计用于完全微调，所有参数都进行了更新，可缓解与过度参数冗余相关的过拟合。因此，LoRA 可训练参数可忽略不计，而以前的 dropout 方法的有效性却被忽视，这可能导致矛盾。为了填补这一空白，我们首先确认参数高效的 LoRA 也容易过拟合。然后我们重新审视特定于变压器的 dropout 方法，并在数学和经验上建立它们的等价性和区别。在此比较分析的基础上，我们引入了一个统一的框架进行全面研究，该框架基于 dropout 位置、结构模式和补偿措施实例化这些方法。通过这个框架，我们揭示了它们在涉及有限可训练参数时的新偏好和性能比较。该框架还使我们能够将最有利的方面融合到一种名为 HiddenKey 的新型 dropout 方法中。大量实验验证了 HiddenKey 在多个模型和任务中的显著优势和充分性，这突出了它是高性能和参数高效 LLM 微调的首选方法。
___
## JORA: JAX Tensor-Parallel LoRA Library for Retrieval Augmented Fine-Tuning
原文：https://aclanthology.org/2024.acl-demos.15/
### abstract
用于基于检索的任务的大型语言模型 (LLM) 的扩展，特别是在检索增强生成 (RAG) 中，面临着严重的内存限制，尤其是在微调大量提示序列时。当前的开源库支持跨多个 GPU 进行全模型推理和微调，但无法适应检索上下文所需的高效参数分布。为了解决这一差距，我们引入了一个新颖的框架，用于 PEFT 兼容的 GPT 模型微调，利用分布式训练。我们的框架独特地利用 JAX 的即时 (JIT) 编译和张量分片来实现高效的资源管理，从而实现加速微调并减少内存需求。这一进步显著提高了微调 LLM 的可扩展性和可行性，即使在 GPU 资源有限的系统上也是如此。我们的实验表明，与使用四个 GPU 的 Hugging Face/DeepSpeed 实现相比，运行时间提高了 12 倍以上，而每个 GPU 消耗的 VRAM 不到一半。
___
## AFLoRA: Adaptive Freezing of Low Rank Adaptation in Parameter Efficient Fine-Tuning of Large Models
原文：https://aclanthology.org/2024.acl-short.16/
### abstract
我们提出了一种新颖的参数高效微调 (PEFT) 方法，称为低秩自适应的自适应冻结 (AFLoRA)。具体来说，对于每个预训练的冻结权重张量，我们添加一条可训练低秩矩阵的并行路径，即向下投影和向上投影矩阵，每个矩阵后面都有一个特征变换向量。基于新颖的冻结分数，我们在微调过程中逐步冻结这些投影矩阵，以减少计算并缓解过度拟合。我们的实验结果表明，我们可以实现最先进的性能，在 GLUE 基准上评估的平均改进高达 0.85%，同时平均可训练参数减少 9.5 倍。在运行时间方面进行比较时，与类似的 PEFT 替代方案相比，AFLoRA 可以产生高达 1.86 倍的改进。除了我们方法的实际效用之外，我们还提供了有关不同模块中 LoRA 路径的可训练性要求以及不同投影矩阵的冻结时间表的见解。
___
## LoRA-Flow: Dynamic LoRA Fusion for Large Language Models in Generative Tasks
原文：https://aclanthology.org/2024.acl-long.695/
### abstract
LoRA 使用轻量级模块为每个下游任务或领域定制大型语言模型 (LLM)，其中不同的学习附加模块代表不同的技能。将现有的 LoRA 组合起来解决新任务可以增强学习到的 LoRA 的可重用性，对于注释数据有限的任务尤其有益。大多数关于 LoRA 组合的先前研究主要依赖于每个涉及的 LoRA 的任务级权重，使得不同的示例和标记共享相同的 LoRA 权重。然而，在生成任务中，不同的标记可能需要不同的技能来管理。以中文数学任务为例，理解问题描述可能更多地依赖于中文 LoRA，而计算部分可能更多地依赖于数学 LoRA。为此，我们提出了 LoRA-Flow，它利用动态权重来调整不同 LoRA 的影响。每一步的权重由具有极少参数的融合门决定，只需 200 个训练示例即可学习。在六个生成任务中进行的实验表明，我们的方法在任务级融合权重方面始终优于基线。这强调了为 LoRA 组合引入动态融合权重的必要性。
___
## MELoRA: Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning
原文：https://aclanthology.org/2024.acl-long.168/
### abstract
参数高效微调 (PEFT) 是一种流行的定制预训练大型语言模型 (LLM) 的方法，尤其是在模型规模和任务多样性增加的情况下。低秩自适应 (LoRA) 基于这样的思想：自适应过程本质上是低维的，即显著的模型变化可以用相对较少的参数来表示。然而，与全参数微调相比，降低秩会遇到特定任务的泛化误差挑战。我们提出了 MELoRA，这是一种小型集成低秩适配器，它使用更少的可训练参数，同时保持更高的秩，从而提供更高的性能潜力。核心思想是冻结原始预训练权重并训练一组仅具有少量参数的小型 LoRA。这可以捕捉到小型 LoRA 之间的显著多样性，从而提高泛化能力。我们对各种 NLP 任务进行了理论分析和实证研究。我们的实验结果表明，与LoRA相比，MELoRA在自然语言理解任务上可训练参数减少了8倍，在指令遵循任务上可训练参数减少了36倍，取得了更好的性能，证明了MELoRA的有效性。
___
## PRoLoRA: Partial Rotation Empowers More Parameter-Efficient LoRA
原文：https://aclanthology.org/2024.acl-long.156/
### abstract
随着大型语言模型 (LLM) 的快速扩展，同时提供大量低秩自适应 (LoRA) 变得越来越不切实际，导致成本高昂，需要更多参数高效的微调方法。在这项工作中，我们引入了部分旋转增强低秩自适应 (PRoLoRA)，这是一种层内共享机制，包含四个基本组件：广播减少、旋转增强、部分共享细化和修正初始化策略。作为 LoRA 的超集，PRoLoRA 保留了其优势，并有效地规避了对等参数共享方法的缺点，具有卓越的模型容量、实际可行性和广泛的适用性。经验实验证明了 PRoLoRA 在特定参数预算和性能目标场景中具有显着更高的参数效率，并且它可扩展到更大的 LLM。值得注意的是，在可训练参数少一倍的情况下，PRoLoRA 在多个指令调整数据集上仍然优于 LoRA。随后，进行了一项消融研究以验证各个组件的必要性，并强调了 PRoLoRA 相对于三种潜在变体的优势。希望明显更高的参数效率可以使 PRoLoRA 成为 LoRA 的资源友好型替代方案。
___
## LoRAMoE: Alleviating World Knowledge Forgetting in Large Language Models via MoE-Style Plugin
原文：https://aclanthology.org/2024.acl-long.106/
### abstract
监督微调 (SFT) 是大型语言模型 (LLM) 的关键步骤，使它们能够与人类指令保持一致并增强其在下游任务中的能力。大幅增加指令数据是使模型与更广泛的下游任务保持一致或显著提高其在特定任务上的性能的直接解决方案。然而，我们发现指令数据的大规模增加会破坏先前存储在 LLM 中的世界知识。为了应对这一挑战，我们提出了 LoRAMoE，这是一个新颖的框架，它引入了几个低秩适配器 (LoRA) 并使用路由器网络将它们集成在一起，就像 Mixture of Experts (MoE) 的插件版本。它冻结了主干模型并迫使一部分 LoRA 专注于利用世界知识来解决下游任务，以减轻世界知识遗忘。实验结果表明，随着指令数据的增加，LoRAMoE 可以显著提高处理下游任务的能力，同时保持存储在 LLM 中的世界知识。我们的代码可在 https://github.com/Ablustrund/LoRAMoE 获得。
___
## Multimodal Instruction Tuning with Conditional Mixture of LoRA
原文：https://aclanthology.org/2024.acl-long.38/
### abstract
多模态大型语言模型 (MLLM) 在不同领域的各种任务中表现出卓越的能力，并且越来越注重提高其对未见过的多模态任务的零样本泛化能力。多模态指令调整已成为一种成功的策略，通过指令对各种多模态任务上的预训练模型进行微调，从而实现零样本泛化。随着 MLLM 的复杂性和规模不断增长，对参数高效的微调方法（如使用最少参数集进行微调的低秩自适应 (LoRA)）的需求变得至关重要。然而，在多模态指令调整中应用 LoRA 会带来任务干扰的挑战，这会导致性能下降，尤其是在处理广泛的多模态任务时。为了解决这个问题，本文介绍了一种将多模态指令调整与条件混合 LoRA (MixLoRA) 相结合的新方法。它通过动态构建针对每个输入实例的独特需求的低秩自适应矩阵来创新 LoRA，旨在减轻任务干扰。在各种多模态评估数据集上的实验结果表明，MixLoRA 不仅在相同甚至更高的排名上优于传统的 LoRA，还证明了其在各种多模态任务中的有效性和适应性。
___
## FOFO: A Benchmark to Evaluate LLMs’ Format-Following Capability
原文：https://aclanthology.org/2024.acl-long.40/
### abstract
本文介绍了 FoFo，这是评估大型语言模型 (LLM) 遵循复杂、特定领域格式的能力的开创性基准，这是将其用作 AI 代理的关键但尚未得到充分研究的能力。尽管 LLM 取得了进步，但现有基准未能充分评估其格式遵循能力。FoFo 通过 AI-Human 协作方法开发了各种现实世界的格式和指令，填补了这一空白。我们对开源（例如 Llama 2、WizardLM）和闭源（例如 GPT-4、PALM2、Gemini）LLM 的评估突出了三个关键发现：开源模型在格式遵循方面明显落后于闭源模型；LLM 的格式遵循性能与其内容生成质量无关；LLM 的格式熟练程度因领域而异。这些见解表明需要对格式遵循技能进行专门调整，并强调了 FoFo 在指导选择特定领域的 AI 代理方面的作用。 FoFo将会公开发布，为推进LLM评估和应用贡献重要工具。
___
## OPEx: A Component-Wise Analysis of LLM-Centric Agents in Embodied Instruction Following
原文：https://aclanthology.org/2024.acl-long.37/
### abstract
具身指令遵循 (EIF) 是具身学习中的一项关键任务，要求代理通过自我中心观察与其环境进行交互以完成自然语言指令。最近的进展表明，在以框架为中心的方法中采用大型语言模型 (LLM) 来提高包括 EIF 在内的具身学习任务的性能的现象激增。尽管做出了这些努力，但对于从视觉感知到动作执行等各种组件对任务性能的影响，仍缺乏统一的理解。为了解决这一差距，我们引入了 OPEx，这是一个全面的框架，它描述了解决具身学习任务所必需的核心组件：观察者、规划者和执行者。通过广泛的评估，我们对每个组件如何影响 EIF 任务性能进行了深入分析。此外，我们通过将多代理设计集成到以 LLM 为中心的架构的规划器组件中，在这个领域进行了创新，进一步提高了任务性能。我们的研究结果表明，以 LLM 为中心的设计显著改善了 EIF 结果，将视觉感知和低级动作执行确定为关键瓶颈，并证明使用多智能体框架增强 LLM 可进一步提高性能。
___
## GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis
原文：https://aclanthology.org/2024.acl-long.30/
### abstract
大型语言模型 (LLM) 面临越狱提示的威胁。现有的检测越狱提示的方法主要是在线审核 API 或微调的 LLM。然而，这些策略通常需要大量且资源密集的数据收集和训练过程。在本研究中，我们提出了 GradSafe，它通过仔细检查 LLM 中安全关键参数的梯度来有效地检测越狱提示。我们的方法基于一个关键的观察：越狱提示与合规响应配对的 LLM 损失梯度在某些安全关键参数上表现出相似的模式。相反，安全提示会导致不同的梯度模式。基于这一观察，GradSafe 分析了提示（与合规响应配对）的梯度，以准确检测越狱提示。我们表明，在未经进一步训练的情况下应用于 Llama-2 的 GradSafe 在检测越狱提示方面优于 Llama Guard——尽管它使用大型数据集进行了广泛的微调。我们对 ToxicChat 和 XSTest 的评估证明了这种优异的性能在零样本和适应场景中都是一致的。源代码可在 https://github.com/xyq7/GradSafe 上找到。
___
## SportsMetrics: Blending Text and Numerical Data to Understand Information Fusion in LLMs
原文：https://aclanthology.org/2024.acl-long.17/
### abstract
大型语言模型在集成各种数据类型（如文本文档和数据库记录）以实现高级分析方面具有巨大潜力。然而，混合文本和数字数据带来了巨大的挑战。LLM 需要处理和交叉引用实体和数字，处理数据不一致和冗余，并开发规划能力，例如构建用于管理复杂数据查询的工作内存。在本文中，我们介绍了四项以体育数据分析为中心的新任务，以评估 LLM 的数字推理和信息融合能力。这些任务包括为 LLM 提供详细的、逐场的体育比赛描述，然后用对抗场景（如新的比赛规则、更长的持续时间、混乱的叙述）挑战它们，并分析比赛摘要中的关键统计数据。我们对 NBA 和 NFL 比赛进行了广泛的实验，以评估 LLM 在这些任务上的表现。我们的基准 SportsMetrics 引入了一种评估 LLM 数字推理和融合技能的新机制。
___
## BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation
原文：https://aclanthology.org/2024.acl-long.7/
### abstract
大型语言模型 (LLM) 的升级在自然语言处理方面取得了令人瞩目的进展，但也带来了重大的部署挑战。权重量化已成为一种广泛采用的解决方案，可以减少内存和计算需求。本文介绍了 BitDistiller，这是一个将量化感知训练 (QAT) 与知识蒸馏 (KD) 协同起来的框架，可提高超低精度（低于 4 位）下 LLM 的性能。具体而言，BitDistiller 首先采用量身定制的非对称量化和裁剪技术来最大限度地保持量化权重的保真度，然后提出一种新颖的置信度感知 Kullback-Leibler 散度 (CAKLD) 目标，以自蒸馏的方式使用，以实现更快的收敛和卓越的模型性能。实证评估表明，BitDistiller 在一般语言理解和复杂推理基准上显著超越了 3 位和 2 位配置中的现有方法。值得注意的是，BitDistiller 被证明更具成本效益，需要的数据和训练资源更少。代码可在 https://github.com/DD-DuDa/BitDistiller 上找到。