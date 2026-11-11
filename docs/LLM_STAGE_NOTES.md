# 캡스톤 작업 안내 (AI 도우미용 시작 문서)

CARLA 0.9.15 (Windows) + carla_garage(PDM-Lite) 위에서 하는 학부 캡스톤.
구조: GRU가 "언제 LLM을 부를지" 판단 → LLM이 WAIT / BYPASS / PROCEED 의도 결정 → PDM-Lite가 실제 운전.
LLM은 핸들·브레이크를 직접 조작하지 않음. 답은 장애물 등록(BYPASS), 정지 유지(WAIT), 브레이크 해제(PROCEED)로 바뀜.

## 먼저 읽을 것
- `capstone_github/README.md`, `capstone_github/docs/PROGRESS.md` : 전체 진행 기록 (GitHub에 공개된 정리본)
- `capstone_github/gru/model/MODEL.md`, `capstone_github/docs/DATA.md` : GRU 학습 설정·성능·데이터

## 실제 작업 위치 (capstone_github 는 복사본)
- `team_code/intervention_agent.py` : GRU 트리거, 낙하물·공사 통로 처리, 카메라 사진, 로그
- `team_code/llm_advisor.py` : LLM 호출 (stub / claude / gemini), 프롬프트 3종 (앞차 / 낙하물 / 공사 통로)
- `scenario_runner_autopilot/srunner/scenarios/gru_scenarios.py` : 시나리오 (고장차, 정지차, 대기열, 정체, 낙하물, 공사 통로)
- `leaderboard/data/bench2drive_split/` : 경로 XML (165_*, r157_*, r174_*, 165_prop_<물체> 등)
- `run_intervene.ps1` : 실험 실행. 결과는 `results/intervention/<경로>_<시각>/events.jsonl` (+ LLM에 보낸 사진)
- 백업 파일(`*_v13_*`, `*_before_*`)은 지우지 말 것. 시나리오 백업을 srunner/scenarios 안에 두면 클래스가 덮어써지므로 `gru/scenario_backups/`에 둘 것.

## 현재 상태 (2026-11)
- GRU: 완료. `gru/out_long_w20s_4hz_final`, 기준 0.99 + 12초 안전장치. 추가 학습 계획 없음.
- 장애물 분류(직접 주행해 보고 정리, 학습 아님): `results/prop_contact/prop_classes.md`
  - 밟아도 됨: ironplank, creasedbox01~03, dirtdebris01 / 피해야 함: box03, barrel, trashcan01, 콘
- LLM: Gemini 무료 API 연결 완료 (결제 없음).
  - 모델: 처음 `gemini-3.5-flash`로 통일했다가 무료 한도(429) 때문에 `gemini-3-flash-preview`로 변경 (11/11, 화살표 시험 2/2 성공, 응답 약 4 s). 작은 모델(flash-lite)은 공사 통로 방향 판단을 못 맞힘(왼쪽만 답함), 큰 모델은 4/4.
    실패(503 등) 시 같은 모델로 2초 뒤 한 번 더, 그래도 실패하면 stub 규칙. 응답 5~12초.
  - 키는 환경변수로만: `$env:GEMINI_API_KEY`. 코드·깃허브에 절대 넣지 말 것 (저장소 공개).
  - CARLA 없이 테스트: `venv\Scripts\python.exe tools\test_gemini.py` → 사진만으로 장애물 7/7 정답.
  - 시뮬 확인(각 1회): 판자 PROCEED, 상자 BYPASS, 대기열 WAIT, 정지차 WAIT×3 후 40초에 BYPASS → 모두 의도대로.
  - 대기열을 고장차로 착각하던 문제는 프롬프트 보강으로 해결 (단서 없이 오래 서 있는 것만으로는 BYPASS 안 함).

## 안전장치 (2026-11-11 추가) - LLM은 제안, 고정 규칙이 최종 결정
- `llm_advisor.guard()`: 낙하물 높이 추정 > 0.35 m면 PROCEED 금지, 확신도 < 0.6이면 행동 변경 안 함, 공사 통로 폭 < 차폭+0.6 m 또는
  마주 오는 차 40 m 안이면 WAIT, 빨간불·앞의 앞차 출발 중이면 BYPASS 금지. 바뀐 경우 로그에 `guard`, `llm_decision`.
- `intervention_agent` 인내 시간 정책 (도로별, 11/11): LLM에 한 번 이상 묻고 앞이 안 움직이면 `policy_bypass`.
  같은 방향 옆 차선이 비었거나 흐르면 20 s (`PATIENCE_MULTI_SEC`), 반대 차선뿐이면 45 s (`PATIENCE_SEC`),
  빨간불/교차로/옆 차선도 정체면 정책 우회 안 함. 지시문에도 "다차선에서 옆 차선 흐르고 20 s 막히면 차선 변경은 정상 운전" 추가.
  → 대기열도 20 s 넘으면 옆 차선으로 옮길 수 있음 (다차선에서는 정상 동작으로 평가).
- 밟고 지나가기 속도 20 km/h (`PROCEED_KMH`, 이전 12).
- LLM 답에 `seen_in_image`(사진에서 본 것) 포함 → 사진을 실제로 보는지 확인용.
- 대기열 시나리오는 2~3대만 (1대짜리는 비현실적이라 제거). `-Perception` 기본값 noisy (물체 이름 숨김, 크기 ±30 %).
- 실행 하나 상세 확인: `venv\Scripts\python.exe tools\trace_run.py <경로이름>` → 입력 글·사진·답·제어·속도 복귀.

## 공사 구간 + 화살표 (2026-11-11)
- CARLA 화살표 트레일러(`static.prop.trafficwarning`, yaw 270 = 정면)는 전광판이 비어 있음 → 시나리오가 디버그 화살표를 그림
  (`sign_arrow` left/right). 카메라 사진에 찍히는 것 확인.
- `NarrowCorridor` layout: centreline(중앙선 통로) / lane(차선 좁아짐) / closed(전부 막힘) / closure(내 차선 폐쇄 + 화살표).
- 통로 계산은 측정값(콘 위치), LLM은 방향(left/center/right/none)만. 지나갈 수 있는 통로가 여러 개면 LLM이 고른 쪽으로,
  없는 쪽을 고르면 WAIT. 같은 방향 차선으로 옮겼으면 다음 교차로 25 m 전까지 그 차선 유지(`CORR_STAY`, 기본 1).
- 165_closure_left/right: 2차선(오른쪽은 도로 끝) → 오른쪽 화살표여도 LLM이 왼쪽(빈 차선)을 고름 = 화살표를 읽었는지 증명 못 함.
- 가운데 차선 도로 찾기: `.\make_closure_routes.ps1` → m<k>_closure_left/right 생성 (양옆이 다 비어서 화살표로만 결정 가능).

## 실행 예시 (PowerShell, C:\carla_garage, CARLA 켠 상태)
```
$env:GEMINI_API_KEY = "<키>"
$env:LLM_TIMEOUT = "60"   # 모델은 코드 기본값(gemini-3.5-flash)
.\run_intervene.ps1 165_stalled,165_queue -Llm gemini -Trigger gru
.\run_intervene.ps1 165_prop_ironplank,165_prop_box03 -Llm gemini -SpeedCap 30 -Perception noisy
```
주의: `-Trigger` 기본값은 rule(10초). 우리 구조대로 하려면 `-Trigger gru`.

## 다음 할 일 후보
1. 위 4개 경로 각 3~5회 반복 → 성공률·응답 시간 정리 (발표용)
2. 공사 통로(r174_corridor)도 Gemini로 확인
3. 교수님 제안: LLM 답을 모아 작은 모델로 학습(distillation) + 신뢰도 낮으면 진짜 LLM
4. 알려진 한계: GRU 입력·물체 감지는 시뮬 정답값 사용, 낙하물 판단에 크기 추정값도 같이 들어감, 상자 우회 때 옆 차선 끼어들기가 거침(PDM-Lite 특성)

## 작업 방식
- 사용자가 PowerShell에서 직접 실행하고 결과를 알려줌. 짧은 실행 선호. 날씨는 항상 맑음 고정.
- 팀원 작업물은 별도 폴더에 있음. 우리 코드와 합치기 전에 먼저 내용을 확인하고 어떻게 합칠지 물어볼 것.
