# GRU + LLM 교착 판단 (CARLA 캡스톤)

자율주행차가 앞에서 막혔을 때 **"지금 LLM을 불러야 하는가"는 가벼운 GRU가 판단**하고,
**"기다릴까, 돌아갈까"는 LLM이 판단**하고, **실제 운전은 결정론적 플래너(PDM-Lite)가 실행**하는 구조입니다.

```
PDM-Lite 주행 (20 Hz) ──▶ GRU 문지기 (매 tick) ──▶ LLM 해석 (가끔) ──▶ PDM-Lite 실행
                              "부를 만한 정지인가?"     WAIT / BYPASS / PROCEED     차선 변경·통과
```

- 시뮬레이터: CARLA 0.9.15 (Windows), 맵 Town05 (학습) · Town04 (처음 보는 맵 테스트)
- 기반 코드: [carla_garage](https://github.com/autonomousvision/carla_garage) (PDM-Lite, MIT License)
- LLM 자리는 현재 **같은 입출력 형식의 대체 응답기(stub)**. Claude API 연결 코드는 `agents/llm_advisor.py`에 있음

## 폴더 구성과 설치 위치

이 저장소는 carla_garage 위에 얹어 쓰는 파일들만 모은 것입니다. carla_garage를 받은 뒤 아래 위치로 복사하세요.

| 이 저장소 | carla_garage 안의 위치 | 내용 |
|---|---|---|
| `scenarios/gru_scenarios.py` | `scenario_runner_autopilot/srunner/scenarios/` (`scenario_runner/`에도) | 직접 만든 시나리오 5종 |
| `scenarios/broken_down_car.py` | `scenario_runner_autopilot/srunner/scenarios/` | 원본 수정본 (배경 차 정리 추가) |
| `scenarios/lane_clear.py` | `scenario_runner_autopilot/srunner/tools/` | `ClearEgoLaneAhead` (자차 앞 같은 차선 배경 차 제거) |
| `agents/recording_agent.py` | `team_code/` | PDM-Lite + 매 tick CSV 기록 |
| `agents/intervention_agent.py` | `team_code/` | 기록 + GRU 호출 판단 + LLM + 플래너 연결, 낙하물·공사 통로 처리 |
| `agents/llm_advisor.py` | `team_code/` | LLM 프롬프트, 대체 응답기, Claude API 호출 |
| `gru/*.py` | `gru/` | 특징 변환, 학습, 평가, 도로별 교차 검증, 비용 분석 |
| `gru/model/` | `gru/out_long_w20s_4hz_final/` | 최종 학습 모델 (`model.pt`, `norm.json`) |
| `routes/*.xml` | `leaderboard/data/bench2drive_split/` | 시나리오를 배치한 루트 |
| `scripts/*.ps1` | 저장소 최상위 | 수집·개입 실험 실행 스크립트 (PowerShell) |
| `tools/*.py` | `tools/` | 결과 요약 리포트 |
| `patches/` | — | carla_garage 원본에서 바꾼 부분 (`patches/NOTES.md`) |

## 시나리오

모두 PDM-Lite에 **장애물로 등록하지 않습니다**. 등록하면 PDM-Lite가 알아서 피하므로, "누군가 해석해서 알려줘야 하는" 상황을 만들기 위해서입니다.

| 시나리오 (XML type) | 종류 | 설명 | 주요 파라미터 |
|---|---|---|---|
| `BrokenDownCar` | 안 풀림 · LLM 필요 | 비상등 + 경고 삼각대가 있는 고장차 | `distance`, `direction` |
| `StalledCar` | 안 풀림 · LLM 필요 | 아무 단서 없는 정지차. 50%는 앞에 차 한 대 더 | `lead_car` (0/1/-1) |
| `QueueRelease` | 스스로 풀림 | 1~3대 대기열이 자차가 선 뒤 일정 시간 후 출발 | `num_cars`, `hold_min`, `hold_max` |
| `QueueRelease` (짧은) | 스스로 풀림 | 1~5초 만에 출발 (`*_shortqueue`) | `hold_min=1`, `hold_max=5` |
| `StopAndGo` | 스스로 풀림 | 앞차가 3~5초 가고 2~8초 서기를 반복하는 정체 | `go_speed` |
| `RoadDebris` | 낙하물 | 차선 위 정적 물체 (흙더미·판자·상자·드럼통). `side_flow=1`이면 옆 차선에 차가 계속 흐름 | `size`, `prop`, `side_flow`, `flow_speed` |
| `NarrowCorridor` | 공사 임시 통로 | 양쪽 가장자리 라바콘 → 중앙선 위 통로만 남음 (지도에 없는 길) | `length`, `width`, `oncoming` |

루트 이름 규칙: `<도로>_<시나리오>`. 도로는 Town05의 `165`, `r157`, `r174`, `r180`(학습·교차 검증)과 Town04의 `t160`, `t164`(처음 보는 맵 테스트). `normal05`는 시나리오 없는 일반 주행(신호 대기 포함), `174_dense`는 반대편 차가 끊임없이 오는 공식 변형입니다.

## GRU 호출 판단기

| 항목 | 내용 |
|---|---|
| 정답(라벨) | 빨간불이 아닌데 **12초 이상 이어질 정지 = 1**, 그 외 = 0. 멈춘 첫 순간부터 1 (`--label long --long-sec 12`) |
| 입력 | CSV 30열 → 특징 34개 (없음 처리, 원-핫, 정규화). **최근 20초를 초당 4번 = 80 × 34** |
| 모델 | GRU 2층 × 32칸 → 확률 1개. 파라미터 약 1만 3천 개 |
| 학습 | BCE + 양성 가중치, Adam 1e-3, 30 epoch, batch 128 |
| 호출 기준 | **GRU ≥ 0.99 + 15m 안 정지차**, 또는 안전장치 **정지 12초 + 빨간불 아님 + 앞 정지차** |

`norm.json`에 정규화 평균·표준편차, 창 길이(`win=80`), 간격(`step=5` tick), 라벨 종류가 들어 있어 `gru_features.OnlineGRU(model_dir)`가 그대로 읽습니다.

### 주요 결과 (대체 응답기 기준)

| 실험 | 결과 |
|---|---|
| 도로별 교차 검증 (학습에 안 쓴 도로, 불러야 할 정지 24 · 풀리는 정지 44) | 규칙 8초: 8.0초 · 놓침 0 · 헛호출 1/44 → **최종 방식: 6.8초 · 놓침 0 · 헛호출 1/44** |
| 165번 도로 통합 실험 | PDM-Lite만: 고장차·정지차 실패 / 우리 구조: 6판 완주, 충돌 0 |
| Town04 처음 보는 맵 15판 | 15/15 예상대로, 충돌 0, 교착은 2초 안에 호출, 공식 시나리오 7판에서 LLM 호출 0 |
| 낙하물 (작은 흙더미 + 옆 차선 꽉 참) | 무조건 회피: 뒤차와 0.8~0.9m로 끼어듦 / 우리: PROCEED, 차선 변경 없음 |
| 공사 임시 통로 (배경 교통 없음) | PDM-Lite만: 콘과 충돌 / 우리: 통로 따라 통과, 콘과 최소 0.8m |

각 1~2판 실행이라 성공률 통계는 다음 과제입니다.

## 실행

```powershell
# 1) 학습용 주행 기록 (CSV → results\features)
.\run_record.ps1 -Ids 165_stalled,165_queue -Reps 3

# 2) GRU 학습 (20초 창, 4Hz, 전체 데이터로 최종 모델)
venv\Scripts\python.exe gru\train_gru.py --label long --win-sec 20 --hz 4 --extra results\features_newroads --final

# 3) 도로별 교차 검증 · 비용 분석
venv\Scripts\python.exe gru\cross_road.py --win-sec 20 --hz 4
venv\Scripts\python.exe gru\tradeoff.py _w20s_4hz

# 4) 개입 실험 (PDM-Lite + GRU + LLM 대체 응답기)
.\run_intervene.ps1 165_broken,165_stalled,165_queue -Trigger gru
.\run_intervene.ps1 165_debris_small_busy -Trigger gru -Perception noisy
.\run_intervene.ps1 r174_corridor -Trigger gru -NoTraffic
venv\Scripts\python.exe tools\final_report.py _final2
```

`run_intervene.ps1` 주요 옵션: `-Trigger rule|gru`, `-Llm stub|claude`, `-GruThresh 0.99`, `-FallbackSec 12`, `-Perception perfect|noisy`, `-DebrisPolicy llm|always_bypass|off`, `-NoTraffic`, `-Suffix`.

## 데이터

수집한 CSV는 용량 때문에 올리지 않았습니다. 어떤 데이터인지는 [`docs/DATA.md`](docs/DATA.md)에 정리했습니다.

## 한계

- LLM은 아직 대체 응답기. 실제 LLM이 사진을 보는 단계는 남아 있음
- 입력이 시뮬레이터의 정확한 정보(PDM-Lite 특권 정보). 실제 센서에서는 신호·비상등·옆 차선 인식이 나빠짐
- 12초 근처의 애매한 정지가 데이터에 적어 GRU 출력이 0/1로 몰림 (확률 보정 필요)
