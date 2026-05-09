# 사용 방법 요약

이 프로젝트는 실제 주문을 대신 실행하지 않습니다. 토스증권 주문은 사용자가 직접 확인하고 입력해야 합니다.

## 공식 자료 기반 리서치

SEC EDGAR 공식 API에서 최신 공시와 주요 XBRL 수치를 가져옵니다. Google은 자동 스크래핑하지 않고, 공식 출처 확인용 검색 링크만 생성합니다.

```powershell
python agent.py --universe mega official-research --limit 2
```

## 빠른 시장 대응 스캔

장중 5분봉 흐름을 훑고, 상위 후보에 대해 SEC 공식 공시 맥락을 함께 확인합니다.

```powershell
python agent.py --universe stock market-pulse --interval 5m --range 1d --limit 10 --research-top 3
```

## 여러 모델 holdout 비교

과거 학습 구간에서 모델을 고르고, 이후 holdout 구간에서만 평가합니다.

```powershell
python agent.py --universe stock compare-models --train-start 2020-01-01 --train-end 2023-12-31 --test-start 2024-01-01 --profiles stable balanced aggressive --cost-bps 5 --slippage-bps 15 --max-mdd 25
```

## 거래 로그가 포함된 백테스트

백테스트 결과에는 누적 그래프, 일별 수익률 그래프, 거래 로그 CSV가 함께 생성됩니다.

```powershell
python agent.py --universe mega backtest --start 2024-01-01 --top 3 --rebalance-interval 21 --long-window 150 --min-momentum 0.08 --max-volatility 0.45 --stop-loss 0.10 --cost-bps 5 --slippage-bps 15
```

출력 표에는 매수/매도일, 종목, 수량, 체결가, 거래금액, 실현손익, 거래별 수익률이 표시됩니다.

## 자동 주문 설계

실제 주문은 공식 API가 있는 브로커에서만 가능합니다. 현재 구조는 Alpaca Trading API용 어댑터와 dry-run 브로커를 제공합니다. 기본은 주문을 보내지 않는 실행 계획 생성입니다.

```powershell
python agent.py --universe mega auto-order-plan --profile stable --start 2023-01-01 --max-mdd 18 --limit 3 --allocation 0.30
python agent.py submit-plan reports\execution_plan_YYYYMMDD_HHMMSS.json --mode dry-run
```

Alpaca paper 주문은 `.env`에 paper API key를 넣은 뒤 아래처럼 실행합니다.

```powershell
python agent.py submit-plan reports\execution_plan_YYYYMMDD_HHMMSS.json --mode paper
```

Live trading은 코드상 잠겨 있습니다. 사용자가 직접 환경변수 `ALPACA_LIVE_TRADING_ENABLED=true`를 설정하고 `--mode live --i-understand-real-money-risk`를 함께 넣어야만 브로커 API 호출이 가능합니다. 이 프로젝트에서 live 명령은 테스트하지 않았고, 실거래 전 paper trading과 브로커 리스크 설정을 먼저 검증해야 합니다.

## 토스증권 수동 주문 체크리스트

토스증권에 자동 주문을 넣지 않습니다. 후보 종목과 확인 항목만 출력합니다.

```powershell
python agent.py --universe stock toss-plan --profile stable --start 2023-01-01 --max-mdd 18 --limit 5
```

## 현실형 검증

T일 종가로 신호를 만들고 T+1 시가에 체결되는 것으로 가정합니다. 거래비용과 슬리피지를 함께 반영합니다.

```powershell
python agent.py --universe stock walk-forward --profile stable --start 2020-01-01 --train-days 730 --test-days 180 --max-mdd 18 --cost-bps 5 --slippage-bps 15
```

## 남은 한계

- 현재 `stock` 유니버스는 완전한 point-in-time 구성 종목 데이터가 아닙니다.
- 상장폐지 종목과 과거 시점별 시총/거래대금 데이터는 별도 데이터 공급원이 필요합니다.
- Yahoo chart 데이터는 빠른 시장 스캔용 보조 데이터입니다. 실거래용 실시간 호가는 토스증권 화면 또는 유료/공식 브로커 데이터로 확인해야 합니다.
