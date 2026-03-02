from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, AssetType, BalanceAllowanceParams, OrderArgs, OrderType, PartialCreateOrderOptions
from py_clob_client.order_builder.constants import BUY

from polymarket_trade.config import TradeConfig


@dataclass
class ExecResult:
    ok: bool
    side: str
    token_id: str
    order_id: str | None
    detail: str


class PolymarketExecutor:
    def __init__(self, cfg: TradeConfig) -> None:
        self.cfg = cfg
        load_dotenv(".env")

        pk = os.getenv("PRIVATE_KEY")
        if not pk:
            raise RuntimeError("缺少 PRIVATE_KEY（.env）")
        sig_type = int(os.getenv("SIGNATURE_TYPE", "0"))
        funder = os.getenv("FUNDER_ADDRESS") or None
        api_key = os.getenv("POLY_API_KEY")
        api_secret = os.getenv("POLY_SECRET")
        api_pass = os.getenv("POLY_PASSPHRASE")
        if not (api_key and api_secret and api_pass):
            raise RuntimeError("缺少 POLY_API_KEY/POLY_SECRET/POLY_PASSPHRASE（.env）")

        creds = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_pass)
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            key=pk,
            creds=creds,
            signature_type=sig_type,
            funder=funder,
        )
        self.sig_type = sig_type

    def get_collateral_balance_usdc(self) -> float:
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, token_id=None, signature_type=self.sig_type)
        bal = self.client.get_balance_allowance(params)
        raw = bal.get("balance") if isinstance(bal, dict) else getattr(bal, "balance", None)
        if raw is None:
            raise RuntimeError("balance field missing in collateral response")
        return float(raw) / 1_000_000.0

    def buy_token_post_only(
        self,
        *,
        token_id: str,
        usdc: float,
        limit_price: float,
        post_only: bool = True,
        cancel_after_sec: int = 5,
    ) -> ExecResult:
        try:
            px = max(0.001, min(0.999, float(limit_price)))
            shares = max(1e-6, float(usdc) / px)
            signed = self.client.create_order(
                OrderArgs(
                    token_id=token_id,
                    price=float(px),
                    size=float(shares),
                    side=BUY,
                ),
                PartialCreateOrderOptions(tick_size="0.01", neg_risk=False),
            )
            resp: dict[str, Any] = self.client.post_order(signed, OrderType.GTC, post_only=bool(post_only))
            order_id = resp.get("orderID") or resp.get("orderId") or resp.get("order_id")
            ok = bool(resp.get("success", True))
            cancel_resp = None
            if order_id:
                import time

                time.sleep(max(0, int(cancel_after_sec)))
                try:
                    cancel_resp = self.client.cancel_orders([str(order_id)])
                except Exception as ce:
                    cancel_resp = {"cancel_error": repr(ce)}
            detail = {"post_order": resp, "cancel_after_sec": int(cancel_after_sec), "cancel_resp": cancel_resp}
            return ExecResult(ok=ok, side="BUY", token_id=token_id, order_id=str(order_id) if order_id else None, detail=str(detail))
        except Exception as e:
            return ExecResult(ok=False, side="BUY", token_id=token_id, order_id=None, detail=repr(e))

