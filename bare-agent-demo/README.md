# BareAgent 视频演示工程

这是一个故意保留逻辑错误的小型商品分页工程，用于演示 BareAgent 如何根据测试失败自主探索代码、定位跨模块契约问题、修改生产代码并重新验证。

> 请不要直接在这份 seed 上录制。按下文只复制 `catalog/` 和 `tests/` 到临时工作区，不要把本 README 复制给 Agent，否则会提前泄露根因和面试讲解思路。

## 案例结构

```text
bare-agent-demo/
├── catalog/
│   ├── __init__.py
│   ├── pagination.py
│   └── service.py
├── tests/
│   └── test_service.py
└── README.md
```

- `pagination.slice_page` 是内部通用工具，契约为从 `0` 开始的 `page_index`。
- `service.list_products` 是对外服务，契约为从 `1` 开始的 `page`。
- 初始实现在两个契约的边界处缺少转换。
- 内部 helper 测试会防止 Agent 通过改变 helper 原有契约来规避真正问题。

## 录制前准备

以下操作应在录屏开始前完成。从 BareAgent 仓库根目录执行：

```bash
# 将凭据导入当前进程，不打印 .env 内容
set -a
source .env
set +a

# 使用已安装 BareAgent 和 pytest 的虚拟环境
source .venv/bin/activate

# 只复制代码和测试，排除本说明文档
demo_dir="$(mktemp -d /private/tmp/bare-agent-demo.XXXXXX)"
cp -R bare-agent-demo/catalog bare-agent-demo/tests "${demo_dir}/"
cd "${demo_dir}"

# 建立本地基线，便于录制后展示最小 diff
git init -q
git add catalog tests
git -c core.hooksPath=/dev/null \
  -c commit.gpgsign=false \
  -c user.name=Candidate \
  -c user.email=candidate@example.invalid \
  commit -qm "demo baseline"

# 避免终端提示符显示用户名、主机名或个人路径
export PS1='demo$ '
clear
```

录制前还应：

- 关闭系统和即时通信通知。
- 关闭终端命令回显或含个人路径的标题栏。
- 确认画面中没有 API Key、`.env`、姓名、本科院校、邮箱或 GitHub 登录信息。
- 将终端字号调整到录制后仍能在 1080p 画面中读清。

## 演示操作

### 1. 展示初始失败

```bash
python -m pytest -q
```

预期：

```text
3 failed, 1 passed
```

手动执行这一次测试是为了让视频观众清晰看到修复前状态。BareAgent 在任务中仍需要自行执行测试。

### 2. 让 Agent 完成任务

```bash
bare-agent --workspace .
```

在 REPL 中输入：

```text
First run the full test suite. Diagnose the pagination failures, fix only production code without modifying tests, then rerun all tests and briefly explain the root cause.
```

理想轨迹为：

```text
run_command
glob_files / read_file
read_file tests
read_file service
read_file pagination
edit_file
run_command
final answer
```

模型调用期间可以将纯等待片段加速 2～4 倍，但应保留工具名、非零退出标记、文件修改和最终验证的先后顺序。

### 3. 展示修复结果

输入 `/exit` 退出 REPL，然后执行：

```bash
python -m pytest -q
git diff -- catalog/service.py
git diff --exit-code -- tests
```

验收标准：

- 最终为 `4 passed`。
- `tests/` 没有任何差异。
- 生产代码只在服务边界完成 `1-based` 到 `0-based` 的转换。
- Agent 最终状态为 `completed`，且最终说明与真实测试结果一致。

## 视频节奏

建议将成片控制在 1 分 45 秒左右，预留余量，不要贴着 2 分钟上限。

| 时间 | 画面 | 讲解重点 |
| --- | --- | --- |
| 0:00–0:12 | 终端和案例结构 | 这是从零实现的 Coding Agent，只获得自然语言任务 |
| 0:12–0:22 | `3 failed, 1 passed` | 建立修复前基线，不提前打开错误代码 |
| 0:22–1:05 | Agent 工具轨迹 | 测试输出是 Observation，Agent 根据它探索调用关系 |
| 1:05–1:22 | `4 passed` 和一行 diff | 修改测试不算修复；最小生产代码修改通过验证 |
| 1:22–1:48 | `agent.py` 和 `tools.py` | Agent Loop、原生 Tool Calling、本地工具执行和终止条件 |
| 1:48–1:55 | 回到 `4 passed` | 待验证状态与完整 Tool Call–Result round |

## 案例体现的设计思想

### 测试作为可执行规格

分页代码没有语法错误，也不会抛出异常。Agent 必须结合测试期望理解业务语义，这能证明 Coding Agent 不只是修复语法错误或根据 traceback 替换类型。

### 契约属于模块边界

内部 helper 与公共服务的页码语义不同。最小且兼容的修复是在边界处转换，而不是改变已有 helper 契约。这为面试中“为什么修这个文件”提供了可辩护的设计依据。

### Action–Observation 闭环

模型不直接操作文件或终端。它输出结构化 Tool Call，Harness 在本地执行并返回 Tool Result。模型根据新 Observation 继续决策，直到返回最终答案或命中终止条件。

### Verification-aware completion

BareAgent 在文件修改后维护待验证文件。无关的 `ls` 或 `git status` 不能代替测试；本案例要求有效测试命令退出码为 `0` 后再完成任务。

### 可解释的取舍

- Context 以完整 turn/round 为单位裁剪，避免 Tool Call 和 Tool Result 失配。
- 当前不做 LLM semantic compaction，避免摘要失真和额外模型调用。
- 最大步数、最大工具数、重复调用检测、模型错误和用户中断防止无限循环。
- Workspace 限制和命令过滤只是 guardrail，不是操作系统级沙箱。

## 面试可能追问

### 为什么在 service 层修复，而不是修改 pagination helper？

公共 API 使用 1-based 页码，内部 helper 使用 0-based 索引。边界层负责转换能保持两端契约，也不会破坏 helper 的其他潜在调用者。

### Tool Call 和 Tool Result 如何进入 Context？

每个 assistant tool call 必须有匹配 `tool_call_id` 的 tool result。BareAgent 以完整 round 保存二者，并在下一次模型请求中重建协议有效的消息序列。

### 为什么不能只看命令退出码 `0`？

BareAgent 还会判断它是否属于可认可的测试命令。`ls` 或 `git status` 即使成功也不是验证证据；pytest 没有收集到测试时通常返回 `5`，也不会被判定为通过。

### 如何防止 Agent 无限调用工具？

模型最终回答是软终止；步数上限、工具数上限、连续重复工具批次、Context 耗尽、模型错误和人工中断提供硬终止。

### Workspace 路径检查是否等于沙箱？

不等于。它能拒绝路径穿越和明显危险命令，但完整隔离仍需要容器、虚拟机或操作系统级沙箱。

## 备用策略

录制前从 seed 重建临时工作区，连续试跑 5 次。正式采用的轨迹应满足：

- 首个关键动作是运行测试。
- 不修改 `tests/**`。
- 最终 `4 passed`。
- 工具调用不超过 10 次。
- 未剪辑的 Agent 运行时间不超过 45 秒。

如果 5 次中成功少于 3 次，使用已经多次真实验证的 `discount-type` 案例作为视频备用任务，不在录制前临时改动 Agent Harness。
