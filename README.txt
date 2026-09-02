仓库：https://github.com/ngoxuantai270-hub/bare-agent

简介
BareAgent 是从零实现的命令行编程智能体，不使用 Agent 框架或服务端文件、代码执行工具。模型负责决策，本地 Harness 负责上下文、工具执行、输出解析、循环终止和错误处理。

运行
需要 Python 3.11+ 和 uv：
1. uv sync --extra dev
2. ./scripts/configure-env.sh
3. uv run bare-agent --workspace /path/to/project

脚本会隐藏输入 API Key，MODEL 和 BASE_URL 默认为 deepseek-v4-flash 和 https://api.deepseek.com。配置写入权限为 600 且已被 Git 忽略的 .env；也可设置 OPENAI_API_KEY、OPENAI_MODEL、OPENAI_BASE_URL。请将 /path/to/project 替换为目标项目。

省略任务参数即进入内存 REPL，支持 /help、/status、/multi、/reset 和 /exit；多行模式用 /send 提交、/cancel 取消。单次任务：uv run bare-agent --workspace /path/to/project "修复失败测试"。

特色功能
内置 read_file、write_file、edit_file、glob_files、search_text、run_command 六个本地工具。Context 按完整 turn/round 裁剪，工具输出限长。文件修改后进入待验证状态，无关命令或失败测试不能解除验证。路径限制在 workspace 内；命令使用 argv、shell=False、超时和敏感变量过滤。

测试
单元与集成测试：uv run pytest --cov=bare_agent。端到端测评：uv run python scripts/run_evals.py --case all。视频的分页契约错误案例位于 bare-agent-demo，操作见该目录 README.md。安全限制不是操作系统级沙箱。
