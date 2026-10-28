# 수집 데이터 (CSV) 설명

실제 CSV 파일은 용량 때문에 저장소에 올리지 않았습니다 (`.gitignore`의 `*.csv`). 이 문서는 어떤 데이터를 어떻게 모았는지만 정리합니다.

## 어떻게 만들었나

- `scripts/run_record.ps1`가 판마다 CARLA를 새로 켜고 루트를 실행합니다.
- `agents/recording_agent.py`(PDM-Lite를 그대로 상속, 운전은 바꾸지 않음)가 **0.05초(20 Hz)마다 한 줄씩** 기록합니다.
- 한 판 = 한 파일: `results/features/<루트>_<날짜>_<시각>.csv`
- 기록하는 정보는 실제 인식 시스템이 내놓는 "물체 목록" 수준(거리·속도·차선·신호)만입니다.

## GRU 학습에 쓴 데이터

| 시나리오 | 종류 | 파일 수 | 줄 수 |
|---|---|---|---|
| broken (고장차) | 안 풀림 | 8 | 6,674 |
| stalled (정지차) | 안 풀림 | 9 | 7,637 |
| queue (대기열 5~20초) | 풀림 | 9 | 6,841 |
| shortqueue (대기열 1~5초) | 풀림 | 5 | 1,584 |
| stopgo (정체) | 풀림 | 6 | 6,694 |
| normal05 (일반 주행, 신호 대기 포함) | 풀림 | 6 | 6,678 |
| **합계** | | **43** | **36,108 (약 30분 주행)** |

- `results/features`: 165번 도로 + normal05 (28개)
- `results/features_newroads`: r157 · r174 · r180 도로 × 5종 (15개)
- 도로별 교차 검증은 165 · r157 · r174 · r180 중 하나를 빼고 학습, 뺀 도로로 평가했습니다 (normal05는 항상 학습에 포함).

개입 실험(`run_intervene.ps1`) 중에 기록된 CSV는 `results/intervention_features`에 따로 저장되며 학습에는 쓰지 않았습니다.

## 열 구성 (30열)

| 묶음 | 열 | 의미 |
|---|---|---|
| 메타 | `tick`, `sim_time_s`, `tag`, `episode_class` | 순번, 시뮬 시간, 루트 이름, 시나리오 종류(P/N). GRU 입력에는 안 씀 |
| 자차 | `ego_speed_kmh`, `lon_accel_ms2`, `yaw_rate_degs`, `stop_duration_s` | 속도, 가감속, 회전율, 연속 정지 시간 (3 km/h 미만) |
| 앞차 | `front_dist_m`, `front_rel_speed_ms`, `front_speed_ms`, `front_width_m`, `front_class`, `front_hazard_on`, `front_stopped_duration_s` | 같은 차선 50m 안 가장 가까운 물체의 거리·상대속도·속도·폭·종류·비상등·정지 시간 |
| 그 앞차 | `front2_dist_m`, `front2_speed_ms` | 같은 차선 두 번째 물체 |
| 왼쪽 차선 | `left_occupied`, `left_nearest_rel_s_m`, `left_rel_speed_ms`, `left_count`, `left_mean_speed_ms` | 같은 방향 옆 차선의 차 유무, 가장 가까운 차 위치·상대속도, 대수, 평균 속도 |
| 오른쪽 차선 | `right_*` (위와 같음) | |
| 신호·도로 | `traffic_light_red`, `traffic_light_dist_m`, `at_junction` | 경로상 다음 신호(80m 안) 빨간불 여부·거리, 교차로 안인지 |

- "없음"은 `999`로 기록
- 같은 차선 = 좌우 1.8m 이내, 옆 차선 = 1.8~5.5m이면서 진행 방향이 같은 차만
- `front_class`: `none` / `vehicle_stopped` / `vehicle_moving` / `pedestrian`

## CSV → GRU 입력

`gru/gru_features.py`의 `featurize`가 30열을 특징 34개로 바꿉니다.

- 거리는 60m(신호 100m)로 자르고 "있음/없음" 표시를 따로 추가
- 가감속 ±10, 회전율 ±90°/s, 정지 시간 60초로 자름
- `front_class` → 원-핫 4칸
- 학습 데이터의 평균·표준편차로 정규화 (`gru/model/norm.json`)
- 최근 20초를 5 tick 간격(4 Hz)으로 뽑아 80 × 34 입력

## 정답(라벨)

`--label long`: 빨간불이 아닌데 **12초 이상 이어질 정지**는 정지가 시작된 순간부터 1, 나머지는 0. 시나리오 종류(`episode_class`)는 라벨에 쓰지 않습니다. 처음 버전(`--label p`, 고장차 판의 정지 = 1)은 지름길 학습 문제가 있어 바꿨습니다.
