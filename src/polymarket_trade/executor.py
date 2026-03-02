from __future__ import annotations

import os
import math
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

    @staticmethod
    def _snap_down(value: float, step: float, *, precision: int = 8) -> float:
        if step <= 0:
            return float(value)
        units = math.floor(float(value) / float(step) + 1e-12)
        return round(units * float(step), precision)

    def _get_market_rules(self, token_id: str) -> tuple[float, float]:
        tick = 0.01
        min_order_size = 5.0
        try:
            book = self.client.get_order_book(str(token_id))
            if isinstance(book, dict):
                raw_tick = book.get("tick_size")
                raw_min = book.get("min_order_size")
                if raw_tick is not None:
                    tick = max(1e-6, float(raw_tick))
                if raw_min is not None:
                    min_order_size = max(1e-6, float(raw_min))
        except Exception:
            # 使用兜底值，避免因公共盘口接口偶发失败而阻断下单
            pass
        return tick, min_order_size

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
            tick_size, min_order_size = self._get_market_rules(token_id)
            px_raw = max(0.001, min(0.999, float(limit_price)))
            px = self._snap_down(px_raw, tick_size, precision=6)
            px = max(tick_size, min(0.999, px))

            # 仓位金额按市场 tick_size 对齐，减少过细精度带来的下单校验问题
            usdc_raw = max(0.0, float(usdc))
            usdc_aligned = self._snap_down(usdc_raw, tick_size, precision=6)
            if usdc_aligned <= 0:
                usdc_aligned = tick_size

            shares = max(1e-6, usdc_aligned / px)
            if shares < min_order_size:
                shares = min_order_size
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
            detail = {
                "tick_size": tick_size,
                "min_order_size": min_order_size,
                "input_usdc": usdc_raw,
                "aligned_usdc": usdc_aligned,
                "price_raw": px_raw,
                "price_aligned": px,
                "shares": shares,
                "post_order": resp,
                "cancel_after_sec": int(cancel_after_sec),
                "cancel_resp": cancel_resp,
            }
            return ExecResult(ok=ok, side="BUY", token_id=token_id, order_id=str(order_id) if order_id else None, detail=str(detail))
        except Exception as e:
            return ExecResult(ok=False, side="BUY", token_id=token_id, order_id=None, detail=repr(e))

