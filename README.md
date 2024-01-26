---
pipeline_tag: text-generation
---
# InternLM 

<div align="center">

<img src="https://github.com/InternLM/InternLM/assets/22529082/b9788105-8892-4398-8b47-b513a292378e" width="200"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">InternLM</font></b>
    <sup>
      <a href="https://internlm.intern-ai.org.cn/">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    <div>&nbsp;</div>
  </div>
  
[![evaluation](https://github.com/InternLM/InternLM/assets/22529082/f80a2a58-5ddf-471a-8da4-32ab65c8fd3b)](https://github.com/internLM/OpenCompass/)

[🤔Reporting Issues](https://github.com/InternLM/InternLM/issues/new)

</div>


## Introduction

InternLM has open-sourced a 7 billion parameter base model and a chat model tailored for practical scenarios. The model has the following characteristics:
- It leverages trillions of high-quality tokens for training to establish a powerful knowledge base.
- It supports an 8k context window length, enabling longer input sequences and stronger reasoning capabilities.
- It provides a versatile toolset for users to flexibly build their own workflows.

## InternLM-7B

### Performance Evaluation

We conducted a comprehensive evaluation of InternLM using the open-source evaluation tool [OpenCompass](https://github.com/internLM/OpenCompass/). The evaluation covered five dimensions of capabilities: disciplinary competence, language competence, knowledge competence, inference competence, and comprehension competence. Here are some of the evaluation results, and you can visit the [OpenCompass leaderboard](https://opencompass.org.cn/rank) for more evaluation results.

| Datasets\Models           |  **InternLM-Chat-7B** |  **InternLM-7B**  |  LLaMA-7B | Baichuan-7B | ChatGLM2-6B | Alpaca-7B | Vicuna-7B |   
| -------------------- | --------------------- | ---------------- | --------- |  --------- | ------------ | --------- | ---------- |  
| C-Eval(Val)          |      53.2             |        53.4       | 24.2      | 42.7       |  50.9       |  28.9     | 31.2     |
| MMLU                 |      50.8             |       51.0        | 35.2*     |  41.5      |  46.0       |  39.7     | 47.3     |
| AGIEval              |      42.5             |       37.6        | 20.8      | 24.6       |  39.0       | 24.1      | 26.4     |
| CommonSenseQA        |      75.2             |      59.5         | 65.0      | 58.8       | 60.0        | 68.7      | 66.7     |
| BUSTM                |      74.3             |       50.6        | 48.5      | 51.3        | 55.0        | 48.8      | 62.5     |
| CLUEWSC              |      78.6             |      59.1         |  50.3     |  52.8     |  59.8     |   50.3    |  52.2     | 
| MATH                 |      6.4            |         7.1        |  2.8       | 3.0       | 6.6       |  2.2      | 2.8       |
| GSM8K                |      34.5           |        31.2        | 10.1       | 9.7       | 29.2      |  6.0      | 15.3  |
|  HumanEval           |      14.0           |        10.4        |   14.0     | 9.2       | 9.2       | 9.2       | 11.0  |
| RACE(High)           |      76.3           |        57.4        | 46.9*      | 28.1      | 66.3      | 40.7      | 54.0  | 

- The evaluation results were obtained from [OpenCompass 20230706](https://github.com/internLM/OpenCompass/) (some data marked with *, which means come from the original papers), and evaluation configuration can be found in the configuration files provided by [OpenCompass](https://github.com/internLM/OpenCompass/). 
- The evaluation data may have numerical differences due to the version iteration of [OpenCompass](https://github.com/internLM/OpenCompass/), so please refer to the latest evaluation results of [OpenCompass](https://github.com/internLM/OpenCompass/).


**Limitations:** Although we have made efforts to ensure the safety of the model during the training process and to encourage the model to generate text that complies with ethical and legal requirements, the model may still produce unexpected outputs due to its size and probabilistic generation paradigm. For example, the generated responses may contain biases, discrimination, or other harmful content. Please do not propagate such content. We are not responsible for any consequences resulting from the dissemination of harmful information.

### Import from ModelScope
To load the InternLM 7B Chat model using Transformers, use the following code:
```python
import torch
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM

model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', revision='v1.0.2')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto",  trust_remote_code=True, torch_dtype=torch.float16)
model = model.eval()
response, history = model.chat(tokenizer, "hello", history=[])
print(response)
response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
print(response)
```

The responses can be streamed using `stream_chat`:

```python
import torch
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM

model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', revision='v1.0.2')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto",  trust_remote_code=True, torch_dtype=torch.float16)

model = model.eval()
length = 0
for response, history in model.stream_chat(tokenizer, "Hello", history=[]):
    print(response[length:], flush=True, end="")
    length = len(response)
```

### Dialogue

You can interact with the InternLM Chat 7B model through a frontend interface by running the following code:
```bash
pip install streamlit==1.24.0
pip install transformers==4.30.2
streamlit run web_demo.py
```
The effect is as follows

![demo](https://github.com/InternLM/InternLM/assets/9102141/11b60ee0-47e4-42c0-8278-3051b2f17fe4)

## Open Source License

The code is licensed under Apache-2.0, while model weights are fully open for academic research and also allow **free** commercial usage. To apply for a commercial license, please fill in the [application form (English)](https://wj.qq.com/s2/12727483/5dba/)/[申请表（中文）](https://wj.qq.com/s2/12725412/f7c1/). For other questions or collaborations, please contact <internlm@pjlab.org.cn>.

## 简介
InternLM ，即书生·浦语大模型，包含面向实用场景的70亿参数基础模型与对话模型 （InternLM-7B）。模型具有以下特点：
- 使用上万亿高质量预料，建立模型超强知识体系；
- 支持8k语境窗口长度，实现更长输入与更强推理体验；
- 通用工具调用能力，支持用户灵活自助搭建流程；

## InternLM-7B

### 性能评测

我们使用开源评测工具 [OpenCompass](https://github.com/internLM/OpenCompass/) 从学科综合能力、语言能力、知识能力、推理能力、理解能力五大能力维度对InternLM开展全面评测，部分评测结果如下表所示，欢迎访问[ OpenCompass 榜单 ](https://opencompass.org.cn/rank)获取更多的评测结果。

| 数据集\模型           |  **InternLM-Chat-7B** |  **InternLM-7B**  |  LLaMA-7B | Baichuan-7B | ChatGLM2-6B | Alpaca-7B | Vicuna-7B |   
| -------------------- | --------------------- | ---------------- | --------- |  --------- | ------------ | --------- | ---------- |  
| C-Eval(Val)          |      53.2             |        53.4       | 24.2      | 42.7       |  50.9       |  28.9     | 31.2     |
| MMLU                 |      50.8             |       51.0        | 35.2*     |  41.5      |  46.0       |  39.7     | 47.3     |
| AGIEval              |      42.5             |       37.6        | 20.8      | 24.6       |  39.0       | 24.1      | 26.4     |
| CommonSenseQA        |      75.2             |      59.5         | 65.0      | 58.8       | 60.0        | 68.7      | 66.7     |
| BUSTM                |      74.3             |       50.6        | 48.5      | 51.3        | 55.0        | 48.8      | 62.5     |
| CLUEWSC              |      78.6             |      59.1         |  50.3     |  52.8     |  59.8     |   50.3    |  52.2     | 
| MATH                 |      6.4            |         7.1        |  2.8       | 3.0       | 6.6       |  2.2      | 2.8       |
| GSM8K                |      34.5           |        31.2        | 10.1       | 9.7       | 29.2      |  6.0      | 15.3  |
|  HumanEval           |      14.0           |        10.4        |   14.0     | 9.2       | 9.2       | 9.2       | 11.0  |
| RACE(High)           |      76.3           |        57.4        | 46.9*      | 28.1      | 66.3      | 40.7      | 54.0  | 

- 以上评测结果基于 [OpenCompass 20230706](https://github.com/internLM/OpenCompass/) 获得（部分数据标注`*`代表数据来自原始论文），具体测试细节可参见 [OpenCompass](https://github.com/internLM/OpenCompass/) 中提供的配置文件。
- 评测数据会因 [OpenCompass](https://github.com/internLM/OpenCompass/) 的版本迭代而存在数值差异，请以 [OpenCompass](https://github.com/internLM/OpenCompass/) 最新版的评测结果为主。

**局限性：** 尽管在训练过程中我们非常注重模型的安全性，尽力促使模型输出符合伦理和法律要求的文本，但受限于模型大小以及概率生成范式，模型可能会产生各种不符合预期的输出，例如回复内容包含偏见、歧视等有害内容，请勿传播这些内容。由于传播不良信息导致的任何后果，本项目不承担责任。

### 通过 ModelScope加载
通过以下的代码加载 InternLM 7B Chat 模型
```python
import torch
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM

model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', revision='v1.0.2')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
# `torch_dtype=torch.float16` 可以令模型以 float16 精度加载，否则 transformers 会将模型加载为 float32，有可能导致显存不足
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto",  trust_remote_code=True, torch_dtype=torch.float16)
model = model.eval()
response, history = model.chat(tokenizer, "hello", history=[])
print(response)
response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
print(response)
```

如果想进行流式生成，则可以使用 `stream_chat` 接口：

```python
import torch
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM

model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', revision='v1.0.2')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto",  trust_remote_code=True, torch_dtype=torch.float16)

model = model.eval()
length = 0
for response, history in model.stream_chat(tokenizer, "Hello", history=[]):
    print(response[length:], flush=True, end="")
    length = len(response)
```

### 通过前端网页对话
可以通过以下代码启动一个前端的界面来与 InternLM Chat 7B 模型进行交互
```bash
pip install streamlit==1.24.0
pip install transformers==4.30.2
streamlit run web_demo.py
```
效果如下 

![效果](https://github.com/InternLM/InternLM/assets/9102141/11b60ee0-47e4-42c0-8278-3051b2f17fe4)

## 开源许可证

本仓库的代码依照 Apache-2.0 协议开源。模型权重对学术研究完全开放，也可申请免费的商业使用授权（[申请表](https://wj.qq.com/s2/12725412/f7c1/)）。其他问题与合作请联系 <internlm@pjlab.org.cn>。(xtuner) 
