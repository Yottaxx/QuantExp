基于深度学习的 A 股量化投研系统
集数据工程、因子挖掘、深度学习建模与策略回测于一体的量化投资研究平台。本项目基于 PyTorch 实现 PatchTST 时序预测模型，结合多因子模型理论，旨在构建一个高鲁棒性、可扩展的 A 股量化交易系统。

核心特性 (System Features)
1. 数据工程 (Data Engineering)
高性能数据流: 采用多线程并发下载架构，支持全市场 5000+ 只股票数据的快速同步。
存储优化: 基于 Parquet 列式存储构建本地数据湖，实现全内存 Panel 数据加载，通过 float32 精度控制优化内存占用。
增量更新: 内置智能日历检查与文件完整性校验，支持每日增量更新与断点续传。
网络鲁棒性: 集成 VPN 轮询控制器，通过 Clash API 实现 IP 自动切换与连接自检，有效规避反爬策略。
2. 因子建模 (Factor Modeling)
多维因子库: 实现了 Barra 风格因子（动量、波动率、流动性）、经典技术指标、WorldQuant Alpha 101 以及学术前沿的高阶矩（偏度、峰度）和微观结构因子（Yang-Zhang 波动率、Roll Spread）。
预处理流水线:
时序处理: 采用 Rolling Z-Score 进行动态标准化，配合 Clip 截断处理，严格避免未来函数。
截面增强: 计算全市场截面排名 (Cross-Sectional Rank) 及相对强弱特征。
正交化: 引入对称正交化 (Symmetric Orthogonalization) 算法，消除因子共线性。
3. 深度学习模型 (Deep Learning)
模型架构: 采用 PatchTST (Time-Series Transformer)，通过 Patching 技术提取长序列特征，并利用 Channel Independence 处理多变量输入。
优化目标: 设计 Hybrid Loss 损失函数 (1 - IC + MSE)，直接优化预测排名的信息系数 (Information Coefficient)，兼顾数值稳定性。
验证机制: 采用严格的时间序列切分 (Time-Series Split) 进行训练与验证，防止数据泄露。
4. 回测与归因 (Backtesting & Attribution)
实盘模拟: 基于 Backtrader 框架，内置 A 股交易费率模型（印花税 0.05%、佣金 0.03%、最低 5 元）。
风控机制: 实现动态流动性限制（单标的持仓不超过成交量 2%）及资金容量管理。
绩效归因: 提供双轨回测（含费/无费）对比，自动计算 Alpha、Beta、Sharpe Ratio、Max Drawdown 及 Rank IC 等核心指标。
系统架构 (Architecture)
graph TD
    subgraph "Data Layer"
        A[AkShare API] -->|Proxy Rotation| B(Concurrent Downloader)
        B --> C[Parquet Data Lake]
        C -->|Incremental Update| B
    end

    subgraph "Factor Layer"
        C --> D[In-Memory Panel Loader]
        D --> E{Alpha Factory}
        E -->|Compute| F[Raw Factors]
        F -->|Rolling Z-Score| G[Time-Series Norm]
        G -->|CS Rank & Orthogonalization| H[Cross-Sectional Norm]
    end

    subgraph "Model Layer"
        H --> I[Dataset Construction]
        I -->|Time-Series Split| J[PatchTST Model]
        J -->|Hybrid Loss| K((Optimization))
    end

    subgraph "Strategy Layer"
        J -->|Inference| L[Daily Scoring]
        L -->|Regime Filter| M[Top-K Selection]
        M -->|Liquidity Constraint| N[Backtest Engine]
        N --> O[Performance Report]
    end


环境配置 (Installation)
1. Python 依赖
pip install torch transformers datasets pandas numpy scipy scikit-learn akshare backtrader matplotlib tqdm requests pysocks pandarallel accelerate


2. 网络代理配置 (必须)
为保证数据获取的稳定性，系统依赖 Clash 客户端进行网络代理管理。请确保 Clash 客户端已运行并配置如下：
代理端口 (Mixed Port): 7890
外部控制端口 (API Port): 49812
密钥 (Secret): 请在 src/vpn_rotator.py 中填入您的 API Secret。
运行指南 (Usage)
所有功能模块通过 main.py 统一调用。
1. 数据同步
首次运行将初始化本地数据湖，后续运行为增量更新。
python main.py --mode download


2. 模型训练
加载全量数据，计算因子，并在训练集上训练模型。
# 标准训练
python main.py --mode train

# 强制重新计算因子 (修改因子逻辑后使用)
python main.py --mode train --force_refresh

# 超参数调整
python main.py --mode train --mse_weight 0.5 --dropout 0.2


3. 历史绩效分析 (可选)
对指定历史区间进行回溯推理，计算 Rank IC 均值、ICIR 及分层回测曲线。
python main.py --mode analysis --start_date 2024-01-01 --end_date 2024-12-31


4. 每日选股推理
基于最新市场数据进行推理，输出 Top-K 推荐列表，并基于最近数据进行验证性回测。
# 默认配置 (100万资金)
python main.py --mode predict

# 自定义资金与持仓数
python main.py --mode predict --cash 500000 --top_k 10


项目结构 (Project Structure)
SotaQuant/
├── data/                   # 数据存储目录 (Parquet)
├── output/                 # 模型权重与分析图表输出
├── src/
│   ├── alpha_lib.py        # [Core] 因子工厂与预处理逻辑
│   ├── factor_ops.py       # [Core] 向量化算子库
│   ├── data_provider.py    # [Core] 数据加载、清洗、截面处理
│   ├── model.py            # [Core] PatchTST 模型定义与 Loss 实现
│   ├── train.py            # 训练流程控制
│   ├── inference.py        # 推理与选股逻辑
│   ├── backtest.py         # 回测引擎与绩效计算
│   ├── analysis.py         # 历史全量分析工具
│   ├── vpn_rotator.py      # 代理轮询控制器
│   └── config.py           # 全局配置参数
├── main.py                 # 程序入口
└── requirements.txt


免责声明
本系统仅用于量化投资策略的研究与验证。
数据说明: 使用的公开数据源可能存在延迟或误差。
模型风险: 历史回测业绩不代表未来表现，模型存在过拟合风险。
实盘风险: 代码未包含实盘交易接口，实际交易需考虑滑点、撤单等复杂情况。
