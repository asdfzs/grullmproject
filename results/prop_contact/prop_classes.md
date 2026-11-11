# 장애물 분류 (시속 30 km, 맑은 낮, PDM-Lite가 모르는 상태로 직진)

## 밟고 지나갈 것 (PROCEED) — 잘 보이고, 실제로 넘어감
| prop | 이름 | 크기 L×W×H (m) | 주행 결과 |
|---|---|---|---|
| static.prop.ironplank | 철판 | 1.45×1.18×0.02 | 충돌 없이 통과 |
| static.prop.creasedbox03 | 펴진 종이상자 | 1.67×0.72×0.02 | 충돌 없이 통과 |
| static.prop.creasedbox01 | 접힌 종이상자 | 0.81×0.72×0.12 | 충돌 없이 통과 |
| static.prop.creasedbox02 | 접힌 종이상자 (큰 것) | 1.63×0.72×0.21 | 아주 약한 접촉(84) 후 통과 |
| static.prop.dirtdebris01 | 흙더미 | 1.85×1.52×0.15 | 충돌 없이 통과 |

## 피할 것 (BYPASS) — 잘 보이고, 부딪히면 멈춤
| prop | 이름 | 크기 L×W×H (m) | 주행 결과 |
|---|---|---|---|
| static.prop.box03 | 상자 | 0.66×0.67×0.65 | 충돌 16,669 → 정지 |
| static.prop.barrel | 드럼통 | 0.46×0.47×0.81 | 충돌 16,232 → 정지 |
| static.prop.trashcan01 | 쓰레기통 | 0.63×0.57×0.85 | 충돌 17,294 → 정지 |
| static.prop.trafficcone01 | 대형 교통 콘 | 0.88×0.88×1.13 | 충돌 16,999 → 정지 |
| static.prop.constructioncone | 공사용 콘 | 0.34×0.35×0.59 | 충돌 17,158 → 정지 |

## 이번 단계에서 뺄 것
| prop | 이유 |
|---|---|
| brokentile01, garbage05, garbage06 | 너무 작아서 카메라에 거의 안 보임 |
| dirtdebris03 | 흙더미 01과 겹침 (약한 접촉 231 후 통과) |
| plasticbag, shoppingbag | 밟아도 되는 물체인데 CARLA에선 고정 물체라 막힘 → 나중에 "닿기 직전 제거" 방식으로 |
| trashbag | 애매함 (내용물에 따라 다름) + 막힘 |
| plantpot05 | 높이 0.28m로 낮아 경계가 애매함 + 막힘 |
