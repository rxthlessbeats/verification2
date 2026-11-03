# 子豪负责的 pipeline 对接与运行说明

## 已完成的对接

程序直接读取队友的 QERA schema 1.1 JSONL，复用 qera 的 reader、selector_view、build_verifier_input 和 token 计数。没有另建一套 Q/E/R/A 格式。QERA 完整、QA 只有问答、QEA 删除推理、QEsA 选择证据四种条件使用同一个模型指令与解码参数，保持原始时间线。

这里的 agent 是检查者/verifier，输入的是已经冻结的 solver 轨迹，不重新调用 solver 做原始任务，不执行轨迹中的工具。真实服务若使用不同协议，请修改 Backend.call；对方必须提供完整的服务 URL、模型名称和请求返回样例。

新增 src/verification2/pipeline.py、__main__.py、tests/test_pipeline.py；修改原来只打印 Hello 的 verification2 入口。原有 qera 数据处理接口保持不变。

## 安装与离线试跑

Python 3.12+。已经 uv sync 的同学无需额外安装依赖：

```bash
uv sync
uv run verification2 demo
uv run verification2 run --input data/pipeline_demo.jsonl --output outputs/my_mock --backend mock
uv run pytest -q
```

新增 pipeline 测试也可以单独运行：`PYTHONPATH=src python -m unittest tests.test_pipeline -v`。

本次验证：6 项新增 pipeline 测试通过；6 个原有 adapter 测试函数直接运行通过；合成数据 16 次 mock 检查和 1 条原有 SWE-bench fixture 的 4 次 dry-run 输入生成均无错误。本环境缺少 pytest 且无法从包源安装，因此没有运行原仓库完整 pytest 套件；本地请执行上面的 uv run pytest -q。真实 HTTP 模型和三个完整数据集尚未验证。

如果不使用 uv，但已有 pydantic 2，可在项目根目录运行：

```bash
PYTHONPATH=src python -m verification2 demo
PYTHONPATH=src python -m verification2 run --input data/pipeline_demo.jsonl --output outputs/my_mock --backend mock
```

PowerShell 对应：先运行 `$env:PYTHONPATH="src"`，再运行 `python -m verification2 ...`，无需前面的 PYTHONPATH=src。demo 文件或 output 文件夹已存在时需换个路径，程序不覆盖。

包内 data/pipeline_demo.jsonl 已生成，可直接跳过 demo 命令。outputs/mock_smoke 是已运行的模拟结果，不是正式结果。

## 队友的完整数据到位后

压缩包里只有平衡样本的 manifest，没有三个完整的 balanced.jsonl。可以在有网络的本机执行队友原来的命令：

```bash
bash scripts/get_data.sh
uv run verification2 run --input data/tau2_bench.balanced.jsonl data/swebench_verified.balanced.jsonl data/agentrewardbench.balanced.jsonl --output outputs/real_data_prepare --backend dry-run --limit 5
```

注意 --limit 是整个输入列表合计的前 N 条，不是每个 benchmark 各 N 条。要逐个检查，请分别运行、分别指定输出文件夹。先 dry-run 查看 requests.jsonl，再对接模型。默认不截断输入；模型上下文不足时调用会报错，需要统一制定长度筛选方案，而不是只截断某个条件。

## 接入真实模型

支持通用 chat-completions 风格 HTTP 请求，不依赖某家 SDK。下面地址和模型都是占位示意，必须换成 Magnus 提供的真实配置：

```bash
export VERIFIER_API_KEY='在本机设置，勿上传'
uv run verification2 run --input data/tau2_bench.balanced.jsonl --output outputs/qwen_pilot --backend http --endpoint http://MODEL_HOST:8000/v1/chat/completions --model ACTUAL_MODEL_NAME --limit 5
```

无密钥的本地服务可以不设置 VERIFIER_API_KEY。PowerShell 使用 `$env:VERIFIER_API_KEY="..."`。服务应返回 choices[0].message.content（JSON 文本，含 verdict 和 reason），可附 usage；程序严格解析，无法解析时记录 error，不猜测判断。运行器不自动重试，避免重复付费或掩盖失败。

每次只发送固定 system 指令和当前条件的 user 输入，无旧历史、conversation ID、label 或 hidden。服务端也须保证无共享会话状态。未知服务协议需对照请求/返回样例适配，当前 HTTP 测试只验证本机模拟服务器，不意味着已经连上 Magnus 的模型。

## 接 Kai 的工作

若是 E/R 分类修正，先用原有接口 `qera apply-overrides` 生成修正后的 JSONL，再将新文件传给 pipeline；不要重写分类器。

若是证据选择，Kai 输出 JSONL，每条：

```json
{"traj_id":"与数据一致的ID","selected_e":[2,5,9]}
```

然后：

```bash
uv run verification2 run --input data/tau2_bench.balanced.jsonl --output outputs/kai_prepare --backend dry-run --selector file --selections kai_selections.jsonl --budget 2048
```

只接受存在且含 E 的整数 seg_id，重复、缺项和超预算都会报错；顺序由原始轨迹恢复。Kai 选择器应只接 selector_view(traj)，其中含 Q、A 和 E 候选，不含标签、hidden 或 R。选择文件仅以 traj_id 索引，要求跨输入文件的 traj_id 唯一；否则分开运行。

没有 Kai 输出时可用 random（默认）、recency、relevance 试跑，不能把它们当作 Jev。本项目真实选择条件统一叫 QEsA，结果中 selector 字段标明具体方法。若要比较三种 baseline，请各跑一次、使用不同 output、保持 budget 和其他设置一致。

## 输出与研究限制

- requests.jsonl：request_id 和实际模型请求，不含标签/hidden，可用于对接。
- results.jsonl：评估端记录，含真实标签、选择 ID、实际响应、usage、错误和各环节耗时；不要把这整个文件传给模型。
- summary.csv：按 benchmark/条件统计错误召回率、错误拒绝率、平衡准确率。只有所有请求条件都成功的共同样本参加比较；无某一类时对应指标留空。正式比较仍需 task 层配对统计与置信区间。
- manifest.json：输入哈希、模型、指令、tokenizer、预算、随机种子、选择文件哈希。不会保存 API 密钥。

mock 固定返回 correct，故意不模拟真实智能。dry-run 不调用模型。两者的准确率均留空，防止被误作研究结果。

每条轨迹运行前重算 segment/spans/call token 数，所有条件使用同一个当前 tokenizer。默认计数器可能退回近似值；正式实验请配置本地的真实模型 tokenizer（QERA_TOKENIZER，必要时固定 QERA_TOKENIZER_REVISION，并安装项目 qwen 可选依赖）。input_tokens 不含 chat template 包装，是输入长度估计；服务 usage 才是实际计费 token。没有 usage 时留空，不编造金额。

预算只约束被选 E 的正文和 E 调用参数，不包含 Q/A、标题及 chat template。整段 E 超预算则跳过，可能没有任何 E 入选。当前 file selector 的离线生成成本不在运行器耗时中；正式报告 Jev 总成本时必须另外提供其生成耗时与 usage 并合计。当前不含 RL 训练。

异常模型返回会记录并继续，进程最终返回非零；输入格式不合法或重复 ID 则在模型调用前停止。中断时已经写出的请求/结果逐条保留；暂不自动续跑。请使用新目录重跑。

## 目前还需要提供

1. 至少一个完整 balanced.jsonl，或者几条脱敏后的真实 QERA 记录，用于最终真实数据接入验证。
2. Magnus 的 endpoint、实际模型名称、一次请求与返回的样例，以及模型上下文长度。密钥只在本机设置。
3. Kai 若已经完成，提供分类 overrides 或 selected_e 文件与预算约定。
