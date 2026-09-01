BareAgent 是一个从零实现的命令行编程智能体，不依赖 Agent 框架，也不使用服务端文件或代码执行工具。模型只负责决策；本地 Harness 自行维护对话、调用工具并控制循环。

环境：Python 3.11+、uv。安装：uv sync --extra dev。仅通过环境变量设置 OPENAI_API_KEY、OPENAI_MODEL；兼容服务可另设 OPENAI_BASE_URL。不得把真实凭据写入仓库、说明或视频。

单任务：uv run bare-agent --workspace ./project "修复失败测试"
内存 REPL：uv run bare-agent --workspace ./project
命令：/help、/status、/reset、/exit。退出后历史不会保存。可用 --trace-jsonl 文件名记录不含任务、模型正文、工具参数及输出的事件元数据。

内置工具：read_file、write_file、edit_file、glob_files、search_text、run_command。路径限制在工作区内；命令使用 argv 且 shell=False，并有超时、输出截断、敏感环境变量过滤和危险命令拦截。上述措施不是操作系统级沙箱，运行不可信代码仍建议使用容器。

验证：uv run --extra dev pytest --cov=bare_agent；uv run --extra dev ruff check .；uv run --extra dev mypy src；uv build。示例缺陷工程位于 examples/bugfix_demo。
