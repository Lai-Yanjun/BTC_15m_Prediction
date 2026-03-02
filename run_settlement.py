from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from web3 import Web3


USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
PROXY_WALLET_FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"
ZERO_BYTES32 = "0x" + "00" * 32

CTF_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "collateralToken", "type": "address"},
            {"internalType": "bytes32", "name": "parentCollectionId", "type": "bytes32"},
            {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
            {"internalType": "uint256[]", "name": "indexSets", "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]

PROXY_FACTORY_ABI = [
    {
        "constant": False,
        "inputs": [
            {
                "components": [
                    {"name": "typeCode", "type": "uint8"},
                    {"name": "to", "type": "address"},
                    {"name": "value", "type": "uint256"},
                    {"name": "data", "type": "bytes"},
                ],
                "name": "calls",
                "type": "tuple[]",
            }
        ],
        "name": "proxy",
        "outputs": [{"name": "returnValues", "type": "bytes[]"}],
        "payable": True,
        "stateMutability": "payable",
        "type": "function",
    }
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Redeem resolved Polymarket condition")
    p.add_argument("--config", default="trade_config.yaml", help="配置文件路径")
    p.add_argument("--condition-id", default="", help="可覆盖配置里的 condition_id")
    p.add_argument("--dry-run", action="store_true", help="仅构造交易不广播")
    p.add_argument("--no-relayer", action="store_true", help="禁用方案A（Relayer），仅走链上直连/Proxy")
    p.add_argument("--proxy", default=None, help="同时设置 HTTP/HTTPS 代理，例如 http://127.0.0.1:7890")
    p.add_argument("--http-proxy", default=None, help="仅设置 HTTP 代理")
    p.add_argument("--https-proxy", default=None, help="仅设置 HTTPS 代理")
    p.add_argument("--no-proxy", default="localhost,127.0.0.1", help="NO_PROXY，默认 localhost,127.0.0.1")
    return p


def _setup_proxy_env(*, proxy: str | None, http_proxy: str | None, https_proxy: str | None, no_proxy: str) -> None:
    from polymarket_trade.proxy_env import inject_proxy_env

    inject_proxy_env(
        proxy=proxy,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        no_proxy=no_proxy,
    )


def _normalize_condition_id(condition_id: str) -> str:
    c = (condition_id or "").strip()
    if not c:
        raise ValueError("condition_id 为空")
    if not c.startswith("0x"):
        c = "0x" + c
    if len(c) != 66:
        raise ValueError("condition_id 长度非法，必须是 32-byte hex")
    return c


def _build_redeem_data(w3: Web3, condition_id: str) -> str:
    ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF), abi=CTF_ABI)
    return ctf.encode_abi(
        "redeemPositions",
        args=[
            Web3.to_checksum_address(USDC_E),
            ZERO_BYTES32,
            condition_id,
            [1, 2],
        ],
    )


def _redeem_via_relayer(
    *,
    private_key: str,
    condition_id: str,
    redeem_data: str,
    dry_run: bool,
) -> dict:
    key = (os.getenv("POLY_BUILDER_API_KEY") or "").strip()
    secret = (os.getenv("POLY_BUILDER_SECRET") or "").strip()
    passphrase = (os.getenv("POLY_BUILDER_PASSPHRASE") or "").strip()
    if not (key and secret and passphrase):
        return {"ok": False, "reason": "MISSING_BUILDER_CREDS"}

    if dry_run:
        return {"ok": True, "route": "relayer_gasless", "dry_run": True}

    try:
        from py_builder_relayer_client.client import RelayClient
        from py_builder_relayer_client.models import OperationType, SafeTransaction
        from py_builder_signing_sdk.config import BuilderApiKeyCreds, BuilderConfig
    except Exception as e:
        return {"ok": False, "reason": f"RELAYER_IMPORT_ERROR:{type(e).__name__}", "detail": repr(e)}

    relayer_url = (os.getenv("POLY_RELAYER_URL") or "https://relayer-v2.polymarket.com").strip()
    try:
        builder_cfg = BuilderConfig(
            local_builder_creds=BuilderApiKeyCreds(
                key=key,
                secret=secret,
                passphrase=passphrase,
            )
        )
        client = RelayClient(relayer_url, 137, private_key, builder_cfg)
        expected_safe = client.get_expected_safe()
        deployed = bool(client.get_deployed(expected_safe))
        deploy_result = None
        if not deployed:
            deploy_resp = client.deploy()
            deploy_result = deploy_resp.wait()
        tx = SafeTransaction(
            to=CTF,
            operation=OperationType.Call,
            data=redeem_data,
            value="0",
        )
        resp = client.execute([tx], metadata=f"Redeem {condition_id[:12]}")
        waited = resp.wait()
        return {
            "ok": True,
            "route": "relayer_gasless",
            "expected_safe": expected_safe,
            "deployed_before": deployed,
            "deploy_result": str(deploy_result) if deploy_result is not None else None,
            "transaction_id": getattr(resp, "transaction_id", None),
            "transaction_hash": getattr(resp, "transaction_hash", None),
            "wait_result": str(waited) if waited is not None else None,
        }
    except Exception as e:
        return {"ok": False, "reason": f"RELAYER_EXEC_ERROR:{type(e).__name__}", "detail": repr(e)}


def _redeem_via_chain(
    *,
    w3: Web3,
    private_key: str,
    sig_type: int,
    condition_id: str,
    redeem_data: str,
    dry_run: bool,
) -> dict:
    acct = w3.eth.account.from_key(private_key)
    from_addr = acct.address
    route = "proxy_factory" if sig_type == 1 else "ctf_direct"
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "route": route,
            "from": from_addr,
            "condition_id": condition_id,
        }

    nonce = w3.eth.get_transaction_count(from_addr)
    gas_price = w3.eth.gas_price
    ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF), abi=CTF_ABI)

    if sig_type == 1:
        factory = w3.eth.contract(address=Web3.to_checksum_address(PROXY_WALLET_FACTORY), abi=PROXY_FACTORY_ABI)
        proxy_call = {
            "typeCode": 1,
            "to": Web3.to_checksum_address(CTF),
            "value": 0,
            "data": redeem_data,
        }
        tx = factory.functions.proxy([proxy_call]).build_transaction(
            {
                "from": from_addr,
                "chainId": 137,
                "nonce": nonce,
                "gasPrice": gas_price,
                "value": 0,
            }
        )
        tx["gas"] = int(factory.functions.proxy([proxy_call]).estimate_gas({"from": from_addr}) * 1.2)
    else:
        tx = ctf.functions.redeemPositions(
            Web3.to_checksum_address(USDC_E),
            ZERO_BYTES32,
            condition_id,
            [1, 2],
        ).build_transaction(
            {
                "from": from_addr,
                "chainId": 137,
                "nonce": nonce,
                "gasPrice": gas_price,
                "value": 0,
            }
        )
        tx["gas"] = int(
            ctf.functions.redeemPositions(
                Web3.to_checksum_address(USDC_E),
                ZERO_BYTES32,
                condition_id,
                [1, 2],
            ).estimate_gas({"from": from_addr})
            * 1.2
        )

    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction).hex()
    rcpt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    ok = int(rcpt.status) == 1
    return {
        "ok": ok,
        "route": route,
        "tx_hash": tx_hash,
        "status": int(rcpt.status),
        "block_number": int(rcpt.blockNumber),
        "condition_id": condition_id,
    }


def _main() -> int:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    from polymarket_trade.config import load_trade_config

    args = _build_parser().parse_args()
    _setup_proxy_env(
        proxy=args.proxy,
        http_proxy=args.http_proxy,
        https_proxy=args.https_proxy,
        no_proxy=str(args.no_proxy),
    )
    cfg = load_trade_config(args.config)
    condition_id = _normalize_condition_id(args.condition_id or cfg.condition_id)
    if cfg.neg_risk:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": "NEG_RISK_NOT_SUPPORTED_IN_V1",
                    "hint": "先用标准 yes/no 市场，neg-risk redeem 下个版本补全。",
                },
                ensure_ascii=False,
            )
        )
        return 2

    load_dotenv(".env")
    pk = (os.getenv("PRIVATE_KEY") or "").strip()
    if not pk:
        raise RuntimeError("缺少 PRIVATE_KEY（.env）")
    sig_type = int((os.getenv("SIGNATURE_TYPE") or "0").strip())
    rpc = (os.getenv("RPC_URL") or "https://polygon-rpc.com").strip()

    w3 = Web3(Web3.HTTPProvider(rpc))
    if (not args.dry_run) and (not w3.is_connected()):
        raise RuntimeError(f"RPC 连接失败: {rpc}")
    redeem_data = _build_redeem_data(w3, condition_id)

    relayer_result = None
    if not args.no_relayer:
        relayer_result = _redeem_via_relayer(
            private_key=pk,
            condition_id=condition_id,
            redeem_data=redeem_data,
            dry_run=bool(args.dry_run),
        )
        if relayer_result.get("ok"):
            print(json.dumps(relayer_result, ensure_ascii=False))
            return 0

    chain_result = _redeem_via_chain(
        w3=w3,
        private_key=pk,
        sig_type=sig_type,
        condition_id=condition_id,
        redeem_data=redeem_data,
        dry_run=bool(args.dry_run),
    )
    if relayer_result is not None and not relayer_result.get("ok"):
        chain_result["relayer_fallback"] = relayer_result
    print(json.dumps(chain_result, ensure_ascii=False))
    return 0 if bool(chain_result.get("ok")) else 3


if __name__ == "__main__":
    raise SystemExit(_main())

