**重要声明：本项目仅用于个人量化研究、软件工程演示与学术交流，不构成任何形式的投资建议、证券推荐、收益承诺或交易依据；模型、数据和回测均可能存在错误，使用者须独立判断并自行承担全部风险与损失。**

**Disclaimer: This project is provided solely for personal quantitative research, software engineering demonstration, and academic exchange. It is not investment advice, a securities recommendation, a promise of returns, or a basis for trading. Models, data, and backtests may contain errors; users must exercise independent judgment and assume all risks and losses.**

# 知序量研（ZhiXu Quant Research）v0.1

知序量研是一个可复现的A股日频横截面预测与模拟回测框架。公开它的目的，是展示数据校验、时间序列训练和模拟交易的工程实现。

本项目仅执行历史研究和模拟交易，不连接券商、不发送真实订单、不调用付费模型API。历史表现不代表未来表现。完整风险边界见[DISCLAIMER.md](DISCLAIMER.md)。

本仓库不发布模型收益、超额收益、准确率、IC、回撤、夏普比率或其他能力评估结果，也不对模型效果作任何明示或暗示。研究口径、参数和限制见[PUBLIC_BASELINE.md](PUBLIC_BASELINE.md)。

## 研究口径

- AKShare为主数据源，Baostock用于独立交叉校验。
- 未复权价格用于模拟成交，后复权价格用于特征和标签。
- 第T日收盘后生成信号，最早按T+1日正式开盘价模拟成交。
- 使用时间序列滚动训练和隔离窗口，不进行随机切分。
- 回测考虑佣金、印花税、过户费、滑点、停牌、涨跌停、T+1和100股整数手。
- 原始数据、信号、计划、订单及成交采用不可变或revision归档。
- 数据质量硬失败时停止生成新模拟信号。

## 环境与安装

要求64位Python 3.12。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

Windows PowerShell使用`.venv\Scripts\python.exe`。`requirements.lock`固定Linux/Python 3.12环境的传递依赖与SHA-256哈希。

## 主要命令

```text
python -m quant fetch
python -m quant validate-data
python -m quant build-dataset
python -m quant train
python -m quant backtest
python -m quant research-rank --config rank_calibrated
python -m quant research-portfolio --config portfolio_baseline_final
python -m quant predict
python -m quant report
```

第一次运行建议先执行不联网的测试：

```bash
python -m pytest -m "not live"
```

带`live`标记的测试会访问免费行情接口，可能受到网络、接口调整和限流影响。

## 知序横截面排序基线（ZhiXu Rank Baseline）参数

- 排序来源：知序横截面排序基线，使用固定候选规则和仅基于开发期的校准。
- 目标持股：20只。
- 每周最多替换：2只。
- 持仓跌出前30名后才具备退出资格。
- 权重：置信度与逆波动率组合。
- 单股最大权重：10%。
- 单行业最大权重：25%。
- 不使用市场状态仓位缩放，不使用次日开盘过滤。

## 数据单位

- 股票代码：`SH600000`、`SZ000001`、`BJ430047`。
- 日期：ISO 8601。
- 价格：人民币元。
- 成交量：股；AKShare返回“手”的接口统一乘100。
- 成交额：人民币元。
- 换手率：小数，例如1%保存为`0.01`。
- 缺失表示上游没有提供，不能自动解释为零。

仓库不分发第三方行情、历史预测、账户记录、模型权重或任何真实交易数据。使用者必须自行确认数据提供方条款及当地法律法规。

## 许可证

源代码按MIT许可证提供。许可证中的无担保及责任限制条款不替代适用法律，也不构成法律意见。
