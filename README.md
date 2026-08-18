**重要声明：本项目仅用于个人量化研究、软件工程演示与学术交流，不构成任何形式的投资建议、证券推荐、收益承诺或交易依据；模型、数据和回测均可能存在错误，使用者须独立判断并自行承担全部风险与损失。**

**Disclaimer: This project is provided solely for personal quantitative research, software engineering demonstration, and academic exchange. It is not investment advice, a securities recommendation, a promise of returns, or a basis for trading. Models, data and backtests may contain errors; users must exercise independent judgment and assume all risks and losses.**

# 知序量研（ZhiXu Quant Research）

[![CI](https://github.com/XXLei777Messi10/zhixu-quant-research/actions/workflows/ci.yml/badge.svg)](https://github.com/XXLei777Messi10/zhixu-quant-research/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

知序量研是一个可复现的 A 股日频横截面预测与模拟回测框架。公开仓库的目标是展示数据校验、时间序列训练、交易成本建模和模拟执行的工程方法。

本仓库只维护公开研究基线，不是生产交易系统，也不连接券商、不发送真实订单、不调用付费模型 API。公开基线与任何生产模型、主账户和内部研究线保持隔离。


## 当前公开模型：v0.2.0

当前公开 Release 对应：

~~~text
配置：configs/rank_buffer_final_accounting_repair.yaml
模型线：rank-buffer-v4-locked-data-accounting-repair
~~~

这一版保留 Top20 周度排序框架，并加入排名缓冲、次日开盘模拟执行、轻量开盘过滤，以及历史成分数据和账户份额核算修复。它是较早的公开研究基线，不是主账户生产模型，也不代表生产模型的镜像或收益表现。

完整说明见 [v0.2.0 Release Note](docs/releases/v0.2.0.md) 和 [PUBLIC_BASELINE.md](PUBLIC_BASELINE.md)。

> **先看这里：5 分钟跑通一个不含未来函数的 A 股日频回测骨架。**
>
> 你会看到：固定合成数据的最小回测、交易成本敏感性对比表，以及每周一篇可复核的 Research Note。示例不下载行情、不需要 API 密钥，也不把模拟结果包装成收益承诺。

## 5 分钟 Quick Start

要求 64 位 Python 3.12。

### Windows PowerShell

~~~powershell
git clone https://github.com/XXLei777Messi10/zhixu-quant-research.git
cd zhixu-quant-research
py -3.12 -m venv .venv
.venv\\Scripts\\python.exe -m pip install --upgrade pip
.venv\\Scripts\\python.exe -m pip install -e ".[dev]"
.venv\\Scripts\\python.exe examples/minimal_synthetic.py
~~~

### macOS / Linux

~~~bash
git clone https://github.com/XXLei777Messi10/zhixu-quant-research.git
cd zhixu-quant-research
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python examples/minimal_synthetic.py
~~~

这个示例使用固定的合成数据，不下载行情，不需要 API 密钥，也不代表任何投资结果。

## 最小示例的预期输出

在当前公开基线代码下，输出结构如下：

~~~text
status: ok
trading_days: 10
filled_trades: 2
orders: 2
live_data: false
~~~

输出中的交易数量只用于验证模拟器确实运行，不是策略表现指标。

## 交易成本敏感性 Demo

这个 Demo 用固定的合成毛损益和换手率，比较零成本、基准成本和压力成本三种假设。它展示的是回测口径如何影响研究结论，不是任何真实策略或市场收益。

~~~bash
python examples/cost_sensitivity.py
~~~

~~~text
ZhiXu transaction-cost sensitivity demo
data_source: deterministic synthetic
status: ok
scenario       gross_pnl_bps  total_cost_bps  net_pnl_bps
zero_cost               76.0             0.0         76.0
base_cost               76.0            28.8         47.2
stress_cost             76.0            57.6         18.4
live_data: false
~~~

完整推导见 [Research Note 0005](docs/research-notes/0005-transaction-cost-sensitivity.md)。

## 运行离线测试

~~~bash
python -m ruff check src tests examples
python -m pytest -m "not live and not slow"
~~~

带有 live 标记的测试会访问免费行情接口，不纳入普通 CI。带有 slow 标记的测试用于较长的合成端到端检查，可按需单独运行。

## 运行研究命令

需要自行准备并确认行情数据后，可以使用：

~~~bash
python -m quant fetch
python -m quant validate-data
python -m quant build-dataset
python -m quant train
python -m quant backtest
python -m quant predict
python -m quant report
~~~

合成数据端到端流程：

~~~bash
python -m quant fetch --synthetic
python -m quant validate-data
python -m quant build-dataset
python -m quant train
python -m quant backtest --max-folds 1
python -m quant predict
python -m quant report
~~~

## 目录结构

~~~text
configs/              公开研究和模拟执行配置
src/quant/            核心代码
src/quant/data/       数据适配、归档和校验
src/quant/backtest/   回测规则、模拟账户和指标
src/quant/execution/  信号、计划和模拟执行
tests/                离线测试与 live 接口契约测试
examples/             最小可运行示例
docs/research-notes/  方法论型 Research Notes
README.md             项目入口
PUBLIC_BASELINE.md    公开基线边界和固定参数
DISCLAIMER.md         风险和责任声明
~~~

## 公开研究口径

- AKShare 为主数据源，Baostock 用于独立交叉校验。
- 未复权价格用于模拟成交，后复权价格用于特征和标签。
- 第 T 日收盘后生成信号，最早按 T+1 日正式开盘价模拟成交。
- 使用时间序列滚动训练和隔离窗口，不进行随机切分。
- 回测考虑佣金、印花税、过户费、滑点、停牌、涨跌停、T+1 和 100 股整数手。
- 原始数据、信号、计划、订单及成交采用不可变或 revision 归档。
- 数据质量硬失败时停止生成新的模拟信号。

完整公开基线参数见 [PUBLIC_BASELINE.md](PUBLIC_BASELINE.md)。

## 数据接口限制

- AKShare 和 Baostock 是外部免费接口，可能限流、改字段、短时不可用或修订历史数据。
- 仓库不分发第三方行情、历史预测、账户记录、模型权重或真实交易数据。
- 不同日期重新拉取的数据不保证完全一致；复现时应记录接口版本、拉取时间、数据截止日和复权口径。
- live 测试只验证上游接口契约，不是稳定的离线单元测试。
- 缺失值表示上游没有提供，不能自动解释为零。
- 使用者必须自行确认数据提供方条款及当地法律法规。

## 常见问题

### 为什么必须使用 Python 3.12？

公开依赖和锁定文件按 Python 3.12 维护。其他版本可能可以运行，但不属于当前 CI 验证范围。

### 为什么 CI 不访问实时行情？

外部接口会受到网络、限流和字段变化影响。CI 只运行离线测试，实时接口契约测试使用 live 标记单独运行。

### 为什么我本地结果和别人不同？

行情可能被修订，数据拉取日期、接口版本、股票池和复权口径也可能不同。先比较数据快照和质量报告，再比较模型输出。

### 公开基线和生产模型有什么关系？

公开仓库只展示较早、可解释、可复现的研究基线，不是生产模型的镜像，也不承诺跟随生产模型更新。

### 为什么 README 不展示收益率？

本项目优先展示研究和工程方法，不把历史回测结果写成收益承诺。研究文章会说明口径、限制和失败实验。

### 如何报告问题？

请使用 GitHub Issue 模板，不要上传 API 密钥、账户数据、私有数据或生产配置。安全问题请阅读 [SECURITY.md](SECURITY.md)。

## 方法论内容

Research Notes 位于 [docs/research-notes](docs/research-notes/)。每篇文章说明数据校验、时间切分、交易成本、反未来函数检查和失败实验，不构成投资建议。

## 许可证

源代码按 MIT 许可证提供。许可证中的无担保及责任限制条款不替代适用法律，也不构成法律意见。

## 构建 Python package

当前仓库已提供标准 Python package 配置。要在本地构建并检查 wheel 与源码包：

~~~bash
python -m pip install -e ".[dev]"
python -m build
python -m twine check dist/*
~~~

构建产物位于 dist/。安装测试建议使用干净的 Python 3.12 虚拟环境：

~~~bash
python -m venv package-smoke-env
package-smoke-env/bin/python -m pip install --no-deps dist/*.whl
package-smoke-env/bin/quant --help
~~~

Windows PowerShell 对应命令为：

~~~powershell
py -3.12 -m venv package-smoke-env
package-smoke-env\Scripts\python.exe -m pip install --no-deps (Get-ChildItem dist\*.whl).FullName
package-smoke-env\Scripts\quant.exe --help
~~~

这里的安装检查只验证公开代码包的可安装性，不代表模型效果，也不会下载行情数据。
