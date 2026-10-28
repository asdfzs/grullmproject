# 최종 GRU 모델 (`out_long_w20s_4hz_final`)

| 항목 | 값 |
|---|---|
| 역할 | "지금 이 정지가 LLM에게 물어볼 만한가" 확률 (0~1) |
| 정답(라벨) | 빨간불이 아닌데 12초 이상 이어질 정지 = 1 (정지 시작부터), 그 외 0 |
| 입력 | 특징 34개 × 80칸 (최근 20초, 5 tick = 0.25초 간격) |
| 구조 | GRU 2층 × hidden 32 → Linear → sigmoid, 파라미터 약 1만 3천 개 |
| 학습 데이터 | 43판, 36,108줄 (165 · r157 · r174 · r180 · normal05, `docs/DATA.md`) — `--final`로 전체 사용 |
| 학습 설정 | BCE + 양성 가중치, Adam 1e-3, 30 epoch, batch 128 |
| 학습 명령 | `python gru/train_gru.py --label long --win-sec 20 --hz 4 --extra results\features_newroads --final` |

## 파일

- `model.pt`: PyTorch state_dict. 구조는 `gru/gru_features.py`의 `make_model`
- `norm.json`: 특징별 평균·표준편차(`mean`, `std`), 창 길이 `win=80`, 간격 `step=5`, 사용 특징 `keep`, 라벨 종류 `label=long`

## 사용

```python
from gru_features import OnlineGRU
g = OnlineGRU("gru/model")      # norm.json + model.pt
p = g.push(row)                 # row = recording_agent의 CSV 한 줄(dict). 매 tick 호출 → 확률
```

`intervention_agent.py`는 `p >= 0.99` 이고 15m 안에 정지차가 있으면 LLM을 부릅니다. 놓치는 경우를 위해 "정지 12초 + 빨간불 아님 + 앞 정지차"면 무조건 부르는 안전장치를 같이 씁니다.

## 성능 (학습에 안 쓴 도로 기준, `results/gru_eval/tradeoff_w20s_4hz.txt`)

| 방식 | 호출까지 | 놓침 (24건 중) | 헛호출 (44건 중) |
|---|---|---|---|
| 규칙 8초 + 앞 정지차 | 8.0초 | 0 | 1 |
| GRU ≥ 0.85 | 6.1초 | 1 | 3 |
| GRU ≥ 0.99 | 7.0초 | 2 | 1 |
| **GRU ≥ 0.99 + 12초 안전장치 (사용 중)** | **6.8초** | **0** | **1** |

주의: 출력이 0과 1 근처로 몰립니다 (기준 0.3~0.99 사이 호출 시간 차이 1.4초). 확률 보정이 되어 있지 않으므로 "0.99 = 99% 확실"로 읽으면 안 됩니다. 기준값을 고른 데이터와 평가 데이터가 같아 결과가 약간 낙관적일 수 있습니다.
