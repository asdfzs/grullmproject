# carla_garage 원본에서 바꾼 파일

| 파일 (carla_garage 기준 경로) | 바꾼 내용 | 여기 있는 것 |
|---|---|---|
| `leaderboard_autopilot/leaderboard/scenarios/route_scenario.py` | `NO_BACKGROUND=1`이면 배경 교통을 만들지 않음 (`run_intervene.ps1 -NoTraffic`) | `route_scenario.diff` (5줄), `route_scenario.py` (전체) |
| `scenario_runner_autopilot/srunner/scenarios/broken_down_car.py` | 경로 모드에서 `ClearEgoLaneAhead`를 같이 실행 (자차 앞 같은 차선의 배경 차 제거) | `../scenarios/broken_down_car.py` (전체) |

나머지 파일은 모두 새로 만든 파일이라 원본 수정이 없습니다.
