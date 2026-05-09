# Tradingbot Swing

Python 기반 시장 데이터 분석 및 전략 검증 프로젝트입니다.

이 저장소는 학습, 연구, 시뮬레이션 목적의 코드입니다. 실제 투자 판단이나 수익 보장을 목적으로 하지 않습니다.

---

## 프로젝트 구조

```text
tradingbot_swing/
├── agent.py
├── pyproject.toml
├── README.md
├── USAGE_KO.md
├── .env.example
├── .gitignore
├── src/
│   └── ai_trading_agent/
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── backtest.py
│       ├── swing.py
│       ├── daytrade.py
│       ├── realtime.py
│       ├── realtime_session.py
│       ├── risk.py
│       ├── optimizer.py
│       └── reporting.py
└── tests/
    └── test_strategy_and_risk.py