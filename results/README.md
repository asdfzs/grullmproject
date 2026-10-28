# 결과 파일

모두 대체 응답기(stub LLM) 기준이고, 시나리오별 1~2판 실행 결과입니다.

## gru_eval — GRU 호출 판단기 평가 (도로별 교차 검증)

도로 4곳(165, r157, r174, r180) 중 하나를 빼고 학습한 모델로 뺀 도로를 평가했습니다. "caught" = 불러야 할 정지(12초 이상) 중 호출한 수, "false alarms" = 풀리는 정지인데 호출한 수.

| 파일 | 내용 |
|---|---|
| `cross_road_w3s_20hz_before_stopgo.txt` | 초기: 3초 창, 정체 데이터 추가 전 (풀리는 정지 14건) |
| `cross_road_w3s_20hz_with_stopgo.txt` | 정체·짧은 대기열 추가 후 3초 창 (풀리는 정지 44건) — 3초로는 부족함을 확인 |
| `cross_road_w20s_4hz.txt` | **최종 설정: 20초 창, 4 Hz** |
| `tradeoff_w20s_4hz.txt` / `.png` | 규칙(T초)과 GRU(기준값 0.3~0.995)의 호출 시간·놓침·헛호출 전체 비교, 헛호출 비용 C별 최선 정책 |

## final_test — 처음 보는 맵(Town04) + 공식 시나리오

`final2_town04_and_official.txt`: 최종 모델(GRU ≥ 0.99 + 12초 안전장치)로 15판. 15/15 예상대로, 충돌 0, 헛호출 5회(모두 WAIT).

## extension — 확장 시나리오 (진행 중)

| 파일 | 내용 |
|---|---|
| `debris_busy_lane.txt` | 옆 차선이 꽉 찬 도로의 낙하물. `_db_always*` = 무조건 회피, `_db_llm*` = 우리 방식, `_db_noisy` = 카메라 흉내 인식. "brake forced" 열은 측정 오류가 있어 신뢰하지 않음 |
| `corridor_r174_events.jsonl` | 공사 임시 통로 판의 이벤트 로그 (콘 감지 → 통로 계산 → BYPASS → 통과, 콘과 최소 0.8m) |
