# Redeem 验证与排障手册

适用场景：
- live 在跑，但你怀疑 `redeem` 没触发或失败
- 需要快速确认是「未触发」还是「触发后报错」
- 需要最小化命令闭环做 dry-run 与实盘验证

---

## 1. 先判断是否进入了自动 redeem 阶段

在日志里检查以下字段：
- `risk_block = MARKET_EXPIRED_WAIT_SETTLEMENT`
- `claim_polled = true`
- `last_claim_result` 不为 `null`

若三者都长期没有出现，说明不是 redeem 执行失败，而是还没进入自动结算阶段。

快速筛选（Linux）：

```bash
rg "MARKET_EXPIRED_WAIT_SETTLEMENT|\"claim_polled\": true|\"last_claim_result\": \\{" reports/live_trade_log.jsonl
```

---

## 2. 前置环境检查（必须）

确认 `.env` 至少包含：
- `PRIVATE_KEY`
- `RPC_URL`（例如 `https://polygon-rpc.com`）

若要走 Relayer（默认方案A），还需：
- `POLY_BUILDER_API_KEY`
- `POLY_BUILDER_SECRET`
- `POLY_BUILDER_PASSPHRASE`

检查命令（不回显具体值）：

```bash
python - <<'PY'
import os
keys = [
    "PRIVATE_KEY","RPC_URL",
    "POLY_BUILDER_API_KEY","POLY_BUILDER_SECRET","POLY_BUILDER_PASSPHRASE"
]
for k in keys:
    v = os.getenv(k)
    print(f"{k}: {'OK' if v else 'MISSING'}")
PY
```

---

## 3. 先做 dry-run（推荐）

### 3.1 指定 condition id 做 dry-run

```bash
python run_settlement.py --config trade_config.yaml --condition-id <CONDITION_ID> --dry-run --proxy http://127.0.0.1:7897
```

判定标准：
- 返回 JSON 中 `ok=true`：链路、参数、签名路径基本可用
- `ok=false`：查看 `reason` 字段定位

### 3.2 强制链上路径（绕过 Relayer）做 dry-run

```bash
python run_settlement.py --config trade_config.yaml --condition-id <CONDITION_ID> --dry-run --no-relayer --proxy http://127.0.0.1:7897
```

用途：
- 区分是 Relayer 凭据问题，还是链上/RPC 问题

---

## 4. 实盘 redeem（谨慎）

先确保 dry-run 成功，再执行：

```bash
python run_settlement.py --config trade_config.yaml --condition-id <CONDITION_ID> --proxy http://127.0.0.1:7897
```

执行后关注：
- 输出中的 `tx_hash`、`status`
- live 日志中下一轮的 `equity_usdc` 是否变化

---

## 5. 常见失败原因与修复

### A) `condition_id 为空` / 长度非法
- 原因：传参为空或不是 32-byte hex（66 长度 `0x...`）
- 修复：从对应到期市场日志复制正确 `condition_id`

### B) `MISSING_BUILDER_CREDS`
- 原因：Relayer 凭据缺失
- 修复：补全 `POLY_BUILDER_*`，或临时用 `--no-relayer`

### C) `RPC 连接失败`
- 原因：`RPC_URL` 不通或代理不通
- 修复：更换 RPC、确认代理端口、重试

### D) `GEO_BLOCK_403` / region restricted
- 原因：当前出口 IP 被风控
- 修复：更换代理节点，确保 live 与 settlement 都带 `--proxy`

---

## 6. 自动 redeem 不触发的典型原因

你当前策略如果开启了 `auto_update_15m_market=true`，系统会频繁切换到新 market。  
在这种情况下，若没有进入某个已到期 market 的结算窗口，`claim_polled` 会一直是 `false`。

这属于「未触发」，不是「执行失败」。

---

## 7. 一条命令快速体检

```bash
python run_settlement.py --config trade_config.yaml --condition-id <CONDITION_ID> --dry-run --proxy http://127.0.0.1:7897 && \
python run_settlement.py --config trade_config.yaml --condition-id <CONDITION_ID> --dry-run --no-relayer --proxy http://127.0.0.1:7897
```

两条都 `ok=true` 后，再做实盘 redeem。

