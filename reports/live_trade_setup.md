# Live 交易接入说明（15m Stacking）

## 文件说明
- `run_live_model.py`：统一入口（live/shadow）
- `trade_config.yaml`：交易参数与风控阈值
- `src/polymarket_trade/signal_model.py`：15m stacking 信号推理
- `src/polymarket_trade/executor.py`：Polymarket 下单执行（post-only GTC，5s 未成交撤单）
- `src/polymarket_trade/runner.py`：收盘前 1 秒触发、阈值触发、风控与日志

## 环境准备
1. 复制 `.env.example` 为 `.env`
2. 填入 API 与私钥
3. 安装依赖：
   - `python -m pip install -r requirements.txt`

## 关键配置
- `trade.token_up` / `trade.token_down`：当前市场 token id（必须填写）
- `signal.upper_threshold`：触发阈值（如 0.60）
- `signal.lower_threshold`：做空/反向触发阈值（如 0.40）
- `trade.order_usdc`：单笔下单金额（USDC）
- `trade.dynamic_sizing`：是否启用动态分仓（建议 `true`）
- `trade.equity_ratio`：固定权益比例（Kelly 固定比例，如 `0.085`）
- `trade.min_order_usdc` / `trade.max_order_usdc`：动态分仓上下限
- `trade.cancel_after_sec`：挂单后多少秒自动撤单（默认 5）
- `paths.model_dir`：固定模型目录（首次缺失会自动训练并落盘，之后固定加载）
- `risk.max_orders_per_hour` / `risk.max_usdc_per_day`：频率与日额度限制
- `market.expiry_utc`：市场到期 UTC 时间；到期后系统停止交易并等待结算后更新权益
- `market.condition_id`：当前市场 `condition_id`（redeem 必填）
- `market.neg_risk`：是否 neg-risk 市场（v1 仅支持 `false`）
- `trade.auto_claim_enabled`：到期后是否自动执行 claim
- `trade.claim_command`：自动 claim 的外部命令
- `trade.claim_poll_seconds`：claim 与余额轮询间隔（建议 `900` 秒）

## 运行
- 仅跑一轮（便于联调）：
  - `python run_live_model.py --run-once`
- 只记录不下单（shadow）：
  - `python run_live_model.py --shadow`
- 持续 live：
  - `python run_live_model.py`
- 树莓派 + 代理（与参考仓库参数一致）：
  - `python run_live_model.py --proxy http://127.0.0.1:7897`
  - 或分别设置：`--http-proxy ... --https-proxy ... --no-proxy localhost,127.0.0.1`

## 输出
- `reports/live_trade_log.jsonl`：逐次信号、触发动作、风控拦截、下单回执

## 运行时序（15m）
- 系统每根 K 线只触发一次
- 触发时点：该根 K 线收盘前 1 秒（例如 `12:14:59`）
- 触发后立即计算信号并按阈值决定是否挂单
- 下单方式：post-only GTC，`cancel_after_sec` 到时自动撤单
- 若配置了 `market.expiry_utc` 且已到期：系统不再下单，按 `claim_poll_seconds` 轮询
- 轮询时先执行 `claim_command`（若启用 `auto_claim_enabled`），再读取余额并更新权益/下单金额

## Redeem 脚本
- 内置脚本：`python run_settlement.py --config trade_config.yaml`
- 默认优先走方案A（Relayer gasless redeem）；失败自动回退方案B（链上 Proxy/CTF redeem）
- `SIGNATURE_TYPE=1` 的回退路由为 Proxy Factory redeem
- 需要 `.env` 里存在 `RPC_URL`（例如 `https://polygon-rpc.com`）
- 方案A需要 `.env` 里存在：
  - `POLY_BUILDER_API_KEY`
  - `POLY_BUILDER_SECRET`
  - `POLY_BUILDER_PASSPHRASE`
- 支持与主程序同样的代理参数：
  - `python run_settlement.py --config trade_config.yaml --proxy http://127.0.0.1:7897 --dry-run`

