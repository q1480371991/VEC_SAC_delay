```
TD3 实现（放在 Resource_allocation_V2I/TD3/）

包含文件：
- network.py         : Actor / Critic 网络定义
- td3_agent.py       : TD3Agent 与 ReplayBuffer，包含 save / load
- config.py          : 超参数与场景参数（可根据需要修改）
- train_td3.py       : 训练入口脚本（完整训练循环 + 日志/绘图）
- README.md          : 本文件

如何运行（建议，从仓库根目录运行）：
1. 将整个目录 Resource_allocation_V2I/TD3/ 放入仓库。
2. 确保项目依赖已安装（PyTorch, numpy, matplotlib, pandas 等），与 SAC 环境要求一致。
3. 运行训练：
   python Resource_allocation_V2I/TD3/train_td3.py

说明：
- train_td3.py 会导入项目中的 Environment3（与 main_train.py 一致），并创建独立的 model 与 log 目录（不会覆盖原来的 SAC 模型）。
- 如果需要将 TD3 与 SAC 做严格对比，请保证 config.py 中的随机种子（如果你使用）和训练步数/episodes 与 SAC 一致。
- 如需把 hyperparam/tau 与 SAC 保持完全相同，把 config.py 中的 tau 改为 0.05（主脚本原值）。
```