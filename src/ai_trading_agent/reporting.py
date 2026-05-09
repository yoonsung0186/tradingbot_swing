from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv

from .models import PortfolioSnapshot, RiskDecision, Signal


def format_signal_table(signals: list[Signal]) -> str:
    if not signals:
        return "No signals were generated."
    rows = [
        "symbol action score price stop take_profit reason",
        "------ ------ ----- ----- ---- ----------- ------",
    ]
    for signal in signals:
        rows.append(
            f"{signal.symbol:6} {signal.action:6} {signal.score:5.2f} "
            f"{signal.price:7.2f} {signal.stop_loss:7.2f} {signal.take_profit:11.2f} "
            f"{signal.reason}"
        )
    return "\n".join(rows)


def format_risk_decisions(decisions: list[RiskDecision]) -> str:
    if not decisions:
        return "No orders reviewed."
    rows = [
        "allowed symbol qty notional reason",
        "------- ------ --- -------- ------",
    ]
    for decision in decisions:
        if decision.order:
            rows.append(
                f"{str(decision.allowed):7} {decision.order.symbol:6} "
                f"{decision.order.qty:3} {decision.order.notional:8.2f} {decision.reason}"
            )
        else:
            rows.append(f"{str(decision.allowed):7} {'-':6} {'-':3} {'-':8} {decision.reason}")
    return "\n".join(rows)


def format_trade_table(trades: list[dict], limit: int = 30) -> str:
    if not trades:
        return "No simulated trades."
    rows = [
        "date       symbol side      qty     price    notional   pnl      pnl_pct reason",
        "---------- ------ ---- -------- -------- --------- -------- ------- --------",
    ]
    for trade in trades[:limit]:
        realized_pnl = trade.get("realized_pnl", "")
        pnl_text = "-" if realized_pnl == "" else f"{float(realized_pnl):.2f}"
        pnl_pct = trade.get("pnl_pct", "")
        pnl_pct_text = "-" if pnl_pct == "" else f"{float(pnl_pct):.2%}"
        rows.append(
            f"{trade['date']:10} {trade['symbol']:6} {trade['side']:4} "
            f"{float(trade['qty']):8.3f} {float(trade['price']):8.2f} "
            f"{float(trade['notional']):9.2f} {pnl_text:8} {pnl_pct_text:7} "
            f"{trade.get('reason', '')}"
        )
    if len(trades) > limit:
        rows.append(f"... {len(trades) - limit} more trades")
    return "\n".join(rows)


def format_result_summary_ko(result: dict, title: str = "시뮬레이션 결과") -> str:
    start_equity = float(result["start_equity"])
    end_equity = float(result["end_equity"])
    profit = end_equity - start_equity
    rows = [
        f"## {title}",
        "",
        "| 항목 | 값 |",
        "| --- | ---: |",
        f"| 시작 총자산 | {_money(start_equity)} |",
        f"| 최종 총자산 | {_money(end_equity)} |",
        f"| 총 손익 | {_money(profit)} |",
        f"| 누적 수익률 | {float(result['total_return']):.2%} |",
        f"| 연환산 수익률(CAGR) | {float(result['cagr']):.2%} |",
        f"| 최대 낙폭(MDD) | {float(result['max_drawdown']):.2%} |",
        f"| 연환산 변동성 | {float(result['annual_volatility']):.2%} |",
        f"| Sharpe | {float(result['sharpe']):.2f} |",
        f"| Calmar | {float(result['calmar']):.2f} |",
        f"| 거래 횟수 | {int(result['trades']):,}회 |",
        f"| 분석 거래일 | {int(result['days']):,}일 |",
    ]
    return "\n".join(rows)


def format_trade_table_ko(trades: list[dict], limit: int = 30) -> str:
    if not trades:
        return "시뮬레이션 거래 내역이 없습니다."
    rows = [
        "## 주요 매수/매도 내역",
        "",
        "| 날짜 | 종목 | 구분 | 수량 | 체결가 | 거래금액 | 실현손익 | 수익률 | 사유 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for trade in trades[:limit]:
        realized_pnl = trade.get("realized_pnl", "")
        pnl_text = "-" if realized_pnl == "" else _money(float(realized_pnl))
        pnl_pct = trade.get("pnl_pct", "")
        pnl_pct_text = "-" if pnl_pct == "" else f"{float(pnl_pct):.2%}"
        rows.append(
            f"| {trade['date']} | {trade['symbol']} | {_side_ko(trade['side'])} | "
            f"{float(trade['qty']):,.3f} | {_money(float(trade['price']))} | "
            f"{_money(float(trade['notional']))} | {pnl_text} | {pnl_pct_text} | "
            f"{_reason_ko(trade.get('reason', ''))} |"
        )
    if len(trades) > limit:
        rows.append(f"| ... | ... | ... | ... | ... | ... | ... | ... | 외 {len(trades) - limit:,}건 |")
    return "\n".join(rows)


def format_daytrade_summary_ko(result: dict, title: str = "단타 시뮬레이션 결과") -> str:
    start_equity = float(result["start_equity"])
    end_equity = float(result["end_equity"])
    profit = end_equity - start_equity
    profit_factor = result.get("profit_factor", 0.0)
    profit_factor_text = "무한대" if profit_factor == float("inf") else f"{float(profit_factor):.2f}"
    rows = [
        f"## {title}",
        "",
        "| 항목 | 값 |",
        "| --- | ---: |",
        f"| 시작 총자산 | {_money(start_equity)} |",
        f"| 최종 총자산 | {_money(end_equity)} |",
        f"| 총 손익 | {_money(profit)} |",
        f"| 누적 수익률 | {float(result['total_return']):.2%} |",
        f"| 최대 낙폭(MDD) | {float(result['max_drawdown']):.2%} |",
        f"| 일수 | {int(result['days']):,}일 |",
        f"| 거래 횟수 | {int(result['trades']):,}회 |",
        f"| 승률 | {float(result.get('win_rate', 0.0)):.2%} |",
        f"| 손익비(Profit Factor) | {profit_factor_text} |",
        f"| 평균 거래 수익률 | {float(result.get('average_trade_return', 0.0)):.2%} |",
        f"| 최고 거래 수익률 | {float(result.get('best_trade_return', 0.0)):.2%} |",
        f"| 최악 거래 수익률 | {float(result.get('worst_trade_return', 0.0)):.2%} |",
    ]
    return "\n".join(rows)


def format_daytrade_trade_table_ko(trades: list[dict], limit: int = 30) -> str:
    if not trades:
        return "단타 시뮬레이션 거래 내역이 없습니다."
    rows = [
        "## 단타 매수/매도 내역",
        "",
        "| 기준일(미국장) | 종목 | 진입(KST) | 청산(KST) | 수량 | 진입가 | 청산가 | 거래금액 | 실현손익 | 수익률 | 청산 사유 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for trade in trades[:limit]:
        rows.append(
            f"| {trade['date']} | {trade['symbol']} | {trade['entry_time']} | {trade['exit_time']} | "
            f"{float(trade['qty']):,.3f} | {_money(float(trade['entry_price']))} | "
            f"{_money(float(trade['exit_price']))} | {_money(float(trade['notional']))} | "
            f"{_money(float(trade['realized_pnl']))} | {float(trade['pnl_pct']):.2%} | "
            f"{_daytrade_reason_ko(trade.get('reason', ''))} |"
        )
    if len(trades) > limit:
        rows.append(f"| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | 외 {len(trades) - limit:,}건 |")
    return "\n".join(rows)


def write_trade_log_csv(
    trades: list[dict],
    output_dir: Path = Path("reports"),
    label: str = "trade_log",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = ["date", "symbol", "side", "qty", "price", "notional", "realized_pnl", "pnl_pct", "reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            writer.writerow({field: trade.get(field, "") for field in fieldnames})
    return path


def write_daytrade_trade_log_csv(
    trades: list[dict],
    output_dir: Path = Path("reports"),
    label: str = "daytrade_trade_log",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "date",
        "symbol",
        "side",
        "entry_time",
        "exit_time",
        "qty",
        "entry_price",
        "exit_price",
        "notional",
        "realized_pnl",
        "pnl_pct",
        "stop_price",
        "take_profit_price",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            writer.writerow({field: trade.get(field, "") for field in fieldnames})
    return path


def format_swing_summary_ko(result: dict, title: str = "스윙 시뮬레이션 결과") -> str:
    start_equity = float(result["start_equity"])
    end_equity = float(result["end_equity"])
    profit = end_equity - start_equity
    profit_factor = result.get("profit_factor", 0.0)
    profit_factor_text = "무한대" if profit_factor == float("inf") else f"{float(profit_factor):.2f}"
    rows = [
        f"## {title}",
        "",
        "| 항목 | 값 |",
        "| --- | ---: |",
        f"| 시작 총자산 | {_money(start_equity)} |",
        f"| 최종 총자산 | {_money(end_equity)} |",
        f"| 총 손익 | {_money(profit)} |",
        f"| 누적 수익률 | {float(result['total_return']):.2%} |",
        f"| 연환산 수익률(CAGR) | {float(result['cagr']):.2%} |",
        f"| 최대 낙폭(MDD) | {float(result['max_drawdown']):.2%} |",
        f"| Sharpe | {float(result['sharpe']):.2f} |",
        f"| Calmar | {float(result['calmar']):.2f} |",
        f"| 청산 거래 수 | {int(result.get('trades', 0)):,}회 |",
        f"| 전체 체결 수 | {int(result.get('transaction_count', 0)):,}회 |",
        f"| 승률 | {float(result.get('win_rate', 0.0)):.2%} |",
        f"| Profit Factor | {profit_factor_text} |",
        f"| 평균 청산 수익률 | {float(result.get('average_trade_return', 0.0)):.2%} |",
        f"| 최고 청산 수익률 | {float(result.get('best_trade_return', 0.0)):.2%} |",
        f"| 최악 청산 수익률 | {float(result.get('worst_trade_return', 0.0)):.2%} |",
    ]
    return "\n".join(rows)


def format_swing_trade_table_ko(trades: list[dict], limit: int = 30) -> str:
    if not trades:
        return "스윙 시뮬레이션 청산 거래 내역이 없습니다."
    rows = [
        "## 스윙 청산 거래 내역",
        "",
        "| 진입일 | 청산일 | 종목 | 수량 | 진입가 | 청산가 | 보유일 | 실현손익 | 수익률 | 청산 사유 | 진입 근거 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for trade in trades[:limit]:
        rows.append(
            f"| {trade['entry_date']} | {trade['exit_date']} | {trade['symbol']} | "
            f"{float(trade['qty']):,.3f} | {_money(float(trade['entry_price']))} | "
            f"{_money(float(trade['exit_price']))} | {int(trade.get('bars_held', 0)):,} | "
            f"{_money(float(trade['realized_pnl']))} | {float(trade['pnl_pct']):.2%} | "
            f"{_swing_reason_ko(trade.get('reason', ''))} | {trade.get('entry_reason', '')} |"
        )
    if len(trades) > limit:
        rows.append(f"| ... | ... | ... | ... | ... | ... | ... | ... | ... | 총 {len(trades) - limit:,}건 더 있음 | ... |")
    return "\n".join(rows)


def write_swing_trade_log_csv(
    trades: list[dict],
    output_dir: Path = Path("reports"),
    label: str = "swing_trade_log",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "entry_date",
        "exit_date",
        "symbol",
        "qty",
        "entry_price",
        "exit_price",
        "notional",
        "realized_pnl",
        "pnl_pct",
        "bars_held",
        "reason",
        "entry_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            writer.writerow({field: trade.get(field, "") for field in fieldnames})
    return path


def write_markdown_report(
    regime_note: str,
    signals: list[Signal],
    decisions: list[RiskDecision],
    snapshot: PortfolioSnapshot,
    report_dir: Path = Path("reports"),
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"daily_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    positions = "\n".join(
        f"- {symbol}: {position.qty} shares @ {position.avg_price:.2f}"
        for symbol, position in snapshot.positions.items()
    )
    if not positions:
        positions = "- none"
    content = f"""# US Stock AI Trading Agent Report

Generated: {datetime.now().isoformat(timespec="seconds")}

## Market Regime

{regime_note}

## Portfolio

- Cash: ${snapshot.cash:,.2f}
- Equity: ${snapshot.equity:,.2f}

## Positions

{positions}

## Signals

```text
{format_signal_table(signals)}
```

## Risk Review

```text
{format_risk_decisions(decisions)}
```
"""
    path.write_text(content, encoding="utf-8")
    return path


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _side_ko(side: str) -> str:
    return {"BUY": "매수", "SELL": "매도"}.get(side, side)


def _reason_ko(reason: str) -> str:
    return {
        "rebalance": "리밸런싱",
        "stop_loss": "손절",
    }.get(reason, reason)


def _daytrade_reason_ko(reason: str) -> str:
    return {
        "stop_loss": "손절",
        "take_profit": "목표가",
        "end_of_day": "장마감 청산",
    }.get(reason, reason)


def _swing_reason_ko(reason: str) -> str:
    return {
        "partial_take_profit": "1차 익절",
        "trailing_stop": "트레일링 청산",
        "stop_loss": "손절",
        "ema_exit": "EMA 이탈",
        "time_exit": "최대 보유기간 도달",
        "overheat_exit": "과열 청산",
        "overheat_pullback_exit": "과열 후 고점 이탈 청산",
        "overheat_atr_exit": "과열 후 ATR 청산",
        "end_of_test": "테스트 종료 청산",
    }.get(reason, reason)
