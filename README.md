# 미국 주식 AI Trading Agent

미국 주식/ETF용 보수적 MVP입니다. 기본값은 **실제 주문을 보내지 않는 로컬 paper simulation**이며, Alpaca paper trading 키가 있을 때만 별도 명령으로 paper 주문을 제출할 수 있습니다.

이 프로젝트는 투자 자문이나 수익 보장을 하지 않습니다. 목적은 리서치 자동화, 전략 검증, 주문 전 리스크 체크, paper trading 연습입니다.

## 기능

- Yahoo Finance chart 데이터 기반 미국 ETF/대형주 시그널 생성
- 데이터 캐시(`data/cache`)로 반복 실행 속도 개선
- 모멘텀, 변동성, 드로다운, 거래량, SPY 상관관계/베타 스캔
- SPY 50일 이동평균 기반 risk-on/risk-off 필터
- 모멘텀, 추세, 거래량 조건으로 후보 선정
- 포지션 크기, 현금, 보유 종목 수 기반 리스크 검토
- VIX/장기 추세/손절/거래비용을 반영한 백테스트
- 수익률, CAGR, MDD, 변동성, 샤프, 칼마 비율 기반 전략 최적화
- 시뮬레이션 CSV와 SVG 그래프 생성
- T일 종가 신호, T+1 시가 체결 방식으로 lookahead 편향 완화
- Yahoo adjusted close 기반으로 OHLC를 보정해 분할/배당 왜곡 완화
- 보수적 슬리피지와 최소 거래대금 필터
- Walk-forward 검증으로 최적화 구간과 평가 구간 분리
- 로컬 paper portfolio 저장
- Markdown 일일 리포트 생성
- Alpaca paper account 확인 및 주문 제출 옵션
- 간단한 히스토리 백테스트

## 빠른 실행

Python 3.10 이상이 필요합니다.

```powershell
cd "C:\Users\SSAFY\Documents\New project\ys"
python agent.py report
```

ETF 유니버스 기준으로 리포트만 생성합니다. 결과 파일은 `reports/` 폴더에 저장됩니다.

```powershell
python agent.py paper
```

승인된 BUY 후보가 있으면 로컬 paper portfolio에 가상 매수로 기록합니다. 상태 파일은 기본적으로 `data/paper_state.json`에 저장됩니다.

```powershell
python agent.py --universe etf backtest --start 2024-01-01
```

ETF 유니버스에 대해 간단한 모멘텀 백테스트를 실행합니다.

```powershell
python agent.py --universe all scan --days 1800 --limit 20
```

더 긴 가격/거래량 데이터를 수집해서 모멘텀, 변동성, 드로다운, SPY 상관관계/베타 기준으로 후보를 정렬합니다.

```powershell
python agent.py --universe all optimize --start 2020-01-01 --max-results 10
```

여러 전략 파라미터를 자동으로 돌려서 손실 대비 수익이 좋은 조합을 찾습니다. 실행 후 `reports/`에 CSV와 SVG 그래프가 저장됩니다.

```powershell
python agent.py --universe all optimize --start 2020-01-01 --max-mdd 18
```

최대 낙폭을 약 18% 이하로 제한한 후보만 찾습니다.

일반 주식만 대상으로 안정형/공격형 프로필을 비교할 수도 있습니다.

```powershell
python agent.py --universe stock optimize --profile stable --start 2020-01-01 --max-mdd 18
python agent.py --universe stock optimize --profile aggressive --start 2020-01-01
```

`optimize`와 `backtest`는 누적 평가액 그래프와 일별 수익률 그래프를 함께 생성합니다.

더 현실적인 검증은 아래처럼 walk-forward로 돌립니다.

```powershell
python agent.py --universe stock walk-forward --profile stable --start 2020-01-01 --train-days 730 --test-days 180 --max-mdd 18 --cost-bps 5 --slippage-bps 15
```

이 명령은 과거 2년 구간에서 파라미터를 고르고, 그 다음 약 6개월 구간에서만 평가합니다.

## 백테스트 주의사항

현재 엔진은 같은 날 종가로 판단하고 같은 종가에 체결하지 않습니다. 신호는 T일 종가 이후 만들어지고, 주문은 T+1 시가에 체결되는 것으로 가정합니다.

그래도 완전한 기관급 point-in-time 백테스트는 아닙니다. `stock` 유니버스는 현재 구성한 대형주 리스트이므로 상장폐지 종목과 과거 특정 시점의 실제 편입/제외 이력을 완전히 반영하지 못합니다. 이 편향을 더 줄이려면 과거 시점별 구성 종목, 상장폐지 포함 가격 데이터, 시점별 시총/거래대금 데이터를 공급해야 합니다.

## 유니버스 선택

```powershell
python agent.py --universe etf report
python agent.py --universe sector scan
python agent.py --universe mega report
python agent.py --universe all report
python agent.py --universe etf --symbols AAPL MSFT NVDA report
```

- `etf`: SPY, QQQ, DIA, IWM, TLT, GLD, SHY
- `sector`: XLK, XLF, XLV 등 미국 섹터/테마 ETF
- `mega`: 미국 대형 기술주/우량주
- `stock`: 미국 개별 대형주 중심 유니버스
- `all`: ETF + 섹터 ETF + 대형주
- `--symbols`: 추가 종목

## Alpaca paper trading 연결

1. `.env.example`을 `.env`로 복사합니다.
2. Alpaca paper trading key를 입력합니다.

```powershell
copy .env.example .env
notepad .env
```

계좌 연결 확인:

```powershell
python agent.py alpaca-account
```

Alpaca paper 주문 미리보기:

```powershell
python agent.py --universe etf alpaca-submit
```

실제로 Alpaca paper 주문을 제출:

```powershell
python agent.py --universe etf alpaca-submit --confirm
```

`--confirm` 없이는 주문을 제출하지 않습니다.

## 테스트

```powershell
python -m unittest discover -s tests -p "test*.py"
```

## 현재 전략 요약

BUY 후보 조건:

- 가격이 20일/50일 이동평균 위
- 20일 이동평균이 50일 이동평균 위
- 현재가가 20일 고점 근처
- 거래량이 최근 평균 대비 증가
- SPY가 50일 이동평균 위이거나 방어형 ETF(TLT, GLD, SHY)

리스크 규칙:

- 1회 거래 위험 예산: 계좌 평가액의 0.5%
- 개별 종목 최대 비중: 10%
- 주문 최대 금액: 2,500달러
- 신규 보유 종목 최대 수: 3개

처음에는 `report`와 `paper`만 충분히 돌려보고, 로그와 리포트가 납득될 때 Alpaca paper 주문으로 넘어가는 흐름을 권장합니다.
