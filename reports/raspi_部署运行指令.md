# 树莓派从拉取到运行（15m 实盘版）

仓库地址（已固定）：
- `https://github.com/Lai-Yanjun/BTC_15m_Prediction`
- 分支：`main`

## 一键部署（推荐）

首次在树莓派执行：

```bash
mkdir -p ~/apps
cd ~/apps
git clone -b main https://github.com/Lai-Yanjun/BTC_15m_Prediction.git predicta-live
cd predicta-live
bash deploy_raspi.sh --proxy http://127.0.0.1:7897
```

不走代理可去掉 `--proxy`：

```bash
bash deploy_raspi.sh
```

脚本会自动完成：
- 安装依赖
- 拉取/更新代码
- 建立 `.venv`
- 安装 `requirements.txt`
- 创建 `.env`（若不存在）
- 注册并启动 `predicta-live` 服务

你要做的配置步骤：
1. 编辑环境变量：`nano ~/apps/predicta-live/.env`
2. 编辑交易参数：`nano ~/apps/predicta-live/trade_config.yaml`
3. 重启服务生效：`sudo systemctl restart predicta-live`

停止服务（你关心的）：

```bash
sudo systemctl stop predicta-live
```

也可用快捷脚本：

```bash
cd ~/apps/predicta-live
bash service_control.sh predicta-live stop
bash service_control.sh predicta-live start
bash service_control.sh predicta-live status
bash service_control.sh predicta-live logs
```

## 0) 系统依赖（首次）

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip jq
```

## 1) 拉取代码

```bash
mkdir -p ~/apps
cd ~/apps
git clone -b main https://github.com/Lai-Yanjun/BTC_15m_Prediction.git predicta-live
cd predicta-live
```

已存在项目时更新：

```bash
cd ~/apps/predicta-live
git fetch origin
git checkout main
git reset --hard origin/main
git pull
```

## 2) 创建虚拟环境并安装依赖

```bash
cd ~/apps/predicta-live
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3) 配置环境变量

```bash
cd ~/apps/predicta-live
cp .env.example .env
nano .env
```

至少要填：
- `PRIVATE_KEY`
- `SIGNATURE_TYPE=1`
- `FUNDER_ADDRESS`
- `RPC_URL`
- `POLY_API_KEY` / `POLY_SECRET` / `POLY_PASSPHRASE`
- `POLY_BUILDER_API_KEY` / `POLY_BUILDER_SECRET` / `POLY_BUILDER_PASSPHRASE`（方案A）

快速检查（不打印密钥明文）：

```bash
python - <<'PY'
from dotenv import dotenv_values
v=dotenv_values(".env")
keys=[
 "PRIVATE_KEY","SIGNATURE_TYPE","FUNDER_ADDRESS","RPC_URL",
 "POLY_API_KEY","POLY_SECRET","POLY_PASSPHRASE",
 "POLY_BUILDER_API_KEY","POLY_BUILDER_SECRET","POLY_BUILDER_PASSPHRASE",
]
for k in keys:
    val=(v.get(k) or "").strip()
    print(f"{k}: {'OK' if val else 'MISSING'} len={len(val)}")
PY
```

## 4) 配置交易参数

```bash
nano trade_config.yaml
```

确认这些字段：
- `market.auto_update_15m_market: true`
- `market.market_slug_prefix: "btc-updown-15m"`
- `market.neg_risk: false`
- `signal.upper_threshold` / `signal.lower_threshold`
- `trade.dynamic_sizing: true`
- `trade.equity_ratio: 0.085`
- `trade.claim_command: "python run_settlement.py --config trade_config.yaml"`
- `trade.claim_poll_seconds: 900`
- `trade.settlement_grace_sec: 7200`
- `model_artifact.enabled: true`（使用 Release 自动下载模型时）
- `model_artifact.url: "https://github.com/<owner>/<repo>/releases/download/<tag>/<zip>"`
- `model_artifact.sha256: "<zip 的 sha256>"`
- `model_artifact.require_optimal: true`

建议值（与你当前策略一致）：
- `signal.upper_threshold: 0.52`
- `signal.lower_threshold: 0.48`

## 5) 联调（先 dry-run）

```bash
source .venv/bin/activate
python run_settlement.py --config trade_config.yaml --dry-run
python run_live_model.py --run-once --shadow
```

若你要强制测试回退链路（不用 Relayer）：

```bash
python run_settlement.py --config trade_config.yaml --dry-run --no-relayer
```

如需代理（和你参考仓库一致）：

```bash
python run_settlement.py --config trade_config.yaml --dry-run --proxy http://127.0.0.1:7897
python run_live_model.py --run-once --shadow --proxy http://127.0.0.1:7897
```

模型使用 Release 自动下载（方案 C）时，首次启动前建议先验证：

```bash
python run_live_model.py --run-once --shadow
```

若本地无模型且 `model_artifact.enabled=true`，程序会先下载并校验模型后再启动。

## 6) 正式运行

```bash
source .venv/bin/activate
python run_live_model.py --proxy http://127.0.0.1:7897
```

日志文件：
- `reports/live_trade_log.jsonl`

## 7) 后台常驻（systemd）

创建服务文件：

```bash
sudo nano /etc/systemd/system/predicta-live.service
```

写入：

```ini
[Unit]
Description=Predicta 15m Live Trader
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/apps/predicta-live
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/pi/apps/predicta-live/.venv/bin/python run_live_model.py --proxy http://127.0.0.1:7897
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable predicta-live
sudo systemctl restart predicta-live
sudo systemctl status predicta-live
```

看日志：

```bash
journalctl -u predicta-live -f
```

## 8) 升级流程（每次发版）

```bash
cd ~/apps/predicta-live
git pull
source .venv/bin/activate
python -m pip install -r requirements.txt
sudo systemctl restart predicta-live
```

## 9) 首跑前 10 项检查清单

1. `git branch --show-current` 显示 `main`
2. `.env` 的 10 个关键字段均为 `OK`（见上面的检查脚本）
3. `SIGNATURE_TYPE=1`
4. `market.neg_risk=false`
5. `trade_config.yaml` 中 `market.auto_update_15m_market=true` 且 `market_slug_prefix=btc-updown-15m`
6. `python run_settlement.py --dry-run` 返回 `ok=true`
7. `python run_live_model.py --run-once --shadow` 正常出日志
8. 代理连通：带 `--proxy` 的 dry-run 不报网络错
9. `reports/live_trade_log.jsonl` 持续写入
10. systemd `status` 为 `active (running)`

## 10) 常用运维命令（直接复制）

```bash
# 查看服务状态
sudo systemctl status predicta-live

# 停止 / 启动 / 重启
sudo systemctl stop predicta-live
sudo systemctl start predicta-live
sudo systemctl restart predicta-live

# 开机自启开关
sudo systemctl enable predicta-live
sudo systemctl disable predicta-live

# 实时日志
journalctl -u predicta-live -f
```
