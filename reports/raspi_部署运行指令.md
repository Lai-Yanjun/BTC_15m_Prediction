# 树莓派从拉取到运行（15m 实盘版）

## 0) 系统依赖（首次）

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

## 1) 拉取代码

```bash
mkdir -p ~/apps
cd ~/apps
git clone <你的仓库地址> predicta-live
cd predicta-live
```

已存在项目时更新：

```bash
cd ~/apps/predicta-live
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

## 4) 配置交易参数

```bash
nano trade_config.yaml
```

确认这些字段：
- `market.condition_id`
- `market.neg_risk: false`
- `market.expiry_utc`
- `trade.token_up` / `trade.token_down`
- `signal.upper_threshold` / `signal.lower_threshold`
- `trade.dynamic_sizing: true`
- `trade.equity_ratio: 0.085`

## 5) 联调（先 dry-run）

```bash
source .venv/bin/activate
python run_settlement.py --config trade_config.yaml --dry-run
python run_live_model.py --run-once --shadow
```

如需代理（和你参考仓库一致）：

```bash
python run_settlement.py --config trade_config.yaml --dry-run --proxy http://127.0.0.1:7897
python run_live_model.py --run-once --shadow --proxy http://127.0.0.1:7897
```

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
