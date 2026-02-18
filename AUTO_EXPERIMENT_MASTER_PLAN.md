# SDS_DialMPC Auto-Experiment Master Plan

최종 업데이트: 2026-02-15  
작성 목적: 지금까지 합의한 실행 전략, 단계별 일정, 자동 수정-재실행 시스템 설계를 한 문서에 통합 정리

---

## 1) 현재 상황 요약

### 1.1 지금 겪는 핵심 문제
- 시뮬레이션에서 Go2가 넘어지는 경우가 잦다.
- 입력 영상의 보행 특성을 충분히 따라하지 못하고 다른 행동이 나온다.
- 수동으로 파라미터를 바꿔가며 재실행하면 반복 속도가 너무 느리다.

### 1.2 해결 전략 (핵심 원칙)
- `코드 직접 수정 빈도 최소화` + `파라미터 자동 탐색 빈도 최대화`.
- 실패를 감으로 판단하지 않고 `정량 지표(metrics.json)`로 판단.
- 실패 유형별 자동 수정 규칙을 둬서 `실행 -> 평가 -> 수정 -> 재실행` 폐루프를 구축.
- MacBook은 개발/스모크 테스트, Ubuntu GPU 서버는 대량 실험 운영으로 역할 분리.

---

## 2) 운영 환경 전략 (MacBook + Server)

### 2.1 역할 분리
| 구분 | MacBook | Ubuntu GPU Server |
|---|---|---|
| 주 역할 | 개발/수정 | 대량 실행/운영 |
| 실행 유형 | smoke test, 소규모 검증 | 장시간 sweep, nightly batch |
| 자원 특성 | CPU/메모리 제한 | CUDA GPU 사용 |
| 책임 산출물 | 코드 커밋, 설정 파일, 분석 노트 | trial 결과, 로그, 리포트 |
| 운영 포인트 | 빠른 반복, 디버깅 | 안정 실행, 재시작(resume), 용량 관리 |

### 2.2 표준 워크플로
1. MacBook에서 코드 변경 후 GitHub push
2. 서버에서 pull 후 sweep 실행
3. 서버에서 결과 생성 (`runs/...`, `results.csv`, `metrics.json`)
4. MacBook으로 결과 회수 후 분석/다음 실험 범위 갱신

---

## 3) 단계별 로드맵 (확정안)

## 3.1 1단계 (2026-02-16 ~ 2026-02-17)
실행 경로/보안/입출력 정리

### 목표
- 로컬/서버 어디서든 재현 가능한 실행 구조 만들기
- 민감정보 하드코딩 제거
- 출력 경로를 명확히 통일

### 작업 항목
- `main_sus.py`
- 절대경로 하드코딩 제거
- `--output-dir` 또는 프로젝트 상대 기본 경로 사용
- `agent_gemini.py`
- API 키 하드코딩 제거 (`GEMINI_API_KEY`만 사용)
- 프롬프트 경로 하드코딩 제거 (프로젝트 상대 경로로 해석)
- `gen_reward_code.py`
- 기본 output을 `dial-mpc/dial_mpc/envs/sds_reward_function.py`로 연결
- `inject_sds_code.py`
- 현재 구조와 맞지 않는 패치 방식이므로 폐기 또는 deprecated 경고 스크립트로 전환

### 완료 기준 (DoD)
- 영상 1개 입력 시 리워드 파일이 목표 경로에 항상 갱신됨
- MacBook/서버 양쪽에서 같은 명령으로 동작
- 코드에 API 키/서버 절대경로 하드코딩이 남아있지 않음

---

## 3.2 2단계 (2026-02-18 ~ 2026-02-20)
“실험 1회 표준화” 구현

### 목표
- 실험 1회 실행의 계약(interface) 고정
- 성공/실패 모두 동일한 결과 포맷 저장

### 작업 항목
- `UnitreeGo2EnvConfig`에 reward 파라미터 필드 추가
- `sds_reward_function.py`의 가중치 하드코딩을 config 기반으로 변경
- 단일 실행 스크립트 구축 (`run_single_experiment.py`)
- 필수 인자: `--config`, `--outdir`, `--seed`, `--reward.*`
- 실행 로그/설정/메트릭 표준 저장

### 1회 실험 표준 산출물
- `cmd.txt`
- `config_used.yaml`
- `metrics.json`
- `stdout.log`
- `stderr.log`

### 완료 기준 (DoD)
- 한 줄 커맨드로 1회 실행 가능
- 실패 시에도 `metrics.json`이 생성되고 `success=false`가 기록됨

---

## 3.3 3단계 (2026-02-21 ~ 2026-02-24)
`run_sweep.py` + `sweep_space.yaml` 구축

### 목표
- trial 자동 반복
- 중단 복구(resume)
- topK 보관
- 대량 실험 운영 안정화

### 작업 항목
- `sweep_space.yaml` 작성
- 분포: `loguniform`, `uniform`, `int_uniform`
- `run_sweep.py` 작성
- 샘플링 -> 실행 -> 평가 -> CSV 적재
- timeout / resume / topK / 로그 저장 구현
- 디스크 관리 옵션 추가

### 완료 기준 (DoD)
- 50~100 trial 연속 실행 성공
- 강제 중단 후 resume 시 중복 없이 이어서 실행

---

## 3.4 4단계 (2026-02-25 ~ 2026-02-28)
야간 자동화 + 분석 리포트

### 목표
- 서버에서 매일 자동 실행
- 아침에 결과 요약 자동 생성

### 작업 항목
- cron 또는 nohup/tmux 기반 nightly 실행
- `analyze_results.py` 작성
- top-10 설정 출력
- 파라미터-성능 상관 요약
- 실패율/timeout 비율 리포트
- 선택: WandB offline 저장

### 완료 기준 (DoD)
- 매일 `results.csv` + 요약 리포트 자동 생성
- 운영자가 아침에 리포트만 보고 다음 탐색 범위 결정 가능

### 3.5 일자별 실행 캘린더 (상세)
#### 2026-02-16
- 경로 하드코딩 제거 시작 (`main_sus.py`, `agent_gemini.py`)
- API 키 참조 방식 `GEMINI_API_KEY`로 통일
- 프롬프트 로딩 경로를 프로젝트 상대경로 기반으로 수정

#### 2026-02-17
- `gen_reward_code.py` 기본 출력 경로 정리
- `inject_sds_code.py` deprecated 처리
- 영상 1개 입력 end-to-end 재현성 테스트

#### 2026-02-18
- `UnitreeGo2EnvConfig` reward 파라미터 항목 정의
- `sds_reward_function.py`를 config 기반 계산으로 변경

#### 2026-02-19
- `run_single_experiment.py` 초안 구현
- outdir 표준 산출물 저장 (`cmd.txt`, `config_used.yaml`, 로그)

#### 2026-02-20
- 실패 케이스 포함 `metrics.json` 강제 저장 검증
- 1회 실행 표준 커맨드 확정

#### 2026-02-21
- `sweep_space.yaml` + `run_sweep.py` 초안 구현
- trial 반복/CSV 적재/timeout 구현

#### 2026-02-22
- `resume`, `topK`, `_best` 보관 로직 구현
- 디스크 정리 옵션 초안 추가

#### 2026-02-23
- `classify_failure.py` 및 실패 라벨링 로직 연결
- `propose_next_params.py` 규칙 기반 교정 연결

#### 2026-02-24
- 50~100 trial 안정성 테스트
- 강제 중단 후 resume 복구 테스트

#### 2026-02-25
- `analyze_results.py` 구현 (top10/실패율/상관 요약)
- 리포트 포맷 고정

#### 2026-02-26
- cron/nohup 자동 실행 세팅
- 로그/결과 경로 점검

#### 2026-02-27
- 1일치 nightly dry-run
- 운영 경고/실패 알림 규칙 정리

#### 2026-02-28
- 운영 인수 기준 점검 (체크리스트 완료)
- v1 운영 전환

---

## 4) “넘어짐/동작 불일치” 자동 대처 시스템 설계

## 4.1 핵심 개념
- 문제를 “파라미터 최적화 과제”로 바꾼다.
- trial마다 실패 원인을 자동 분류하고 다음 trial 파라미터를 자동 교정한다.
- 코드 파일을 매 trial마다 직접 수정하지 않는다.

## 4.2 자동 루프 구성
1. `run_single_experiment.py`: 1회 실행 + 표준 파일 저장
2. `classify_failure.py`: 결과/로그 기반 실패 라벨링
3. `propose_next_params.py`: 라벨 기반 파라미터 수정안 생성
4. `run_sweep.py`: 전체 반복 제어
5. `analyze_results.py`: 성능 리포트

## 4.3 표준 trial 폴더 구조
```text
runs/
  sweep_2026-02-21/
    exp_000001/
      cmd.txt
      config_used.yaml
      metrics.json
      stdout.log
      stderr.log
      failure_label.json
    exp_000002/
      ...
  sweep_2026-02-22/
    ...
```

## 4.4 자동 수정-재실행 알고리즘 (의사코드)
```text
for trial in trials:
  params = sample_or_update_params(history, policy_rules)
  run_result = run_single_experiment(params, timeout)
  metrics = load_metrics_or_make_failure(run_result)
  label = classify_failure(metrics, stdout, stderr)
  score = compute_score(metrics)
  save_trial_artifacts(params, metrics, label, score)
  update_topk(score, trial_dir)
  append_results_csv(...)
  if stop_condition_met:
    break
```

## 4.5 stop condition 권장 기준
- `max_trials` 도달
- `wall_time_budget_sec` 초과
- 최근 N개 trial 개선폭이 임계치 미만
- 목표 score 또는 목표 `contact_match_score` 달성

---

## 5) 지표 설계 (metrics.json 스키마)

## 5.1 최소 필수 필드
```json
{
  "success": false,
  "fail_reason": "early_fall",
  "returncode": 1,
  "wall_time_sec": 52.7,
  "fall_time_sec": 1.3,
  "fall_count": 1,
  "mean_roll": 0.42,
  "mean_pitch": 0.35,
  "mean_height_error": 0.08,
  "mean_vel_error": 0.44,
  "yaw_rate_error": 0.11,
  "contact_match_score": 0.51,
  "energy": 63.2,
  "episode_return": -12.7
}
```

## 5.2 목적
- `넘어짐`과 `불일치`를 정량화
- 자동 분류/자동 수정의 입력 데이터로 사용
- trial 간 성능 추세 비교

---

## 6) 실패 분류 규칙과 자동 수정 정책

## 6.1 실패 라벨 정의
- `early_fall`
- `unstable`
- `standstill`
- `wrong_gait`
- `high_energy`
- `compute_timeout`

## 6.2 라벨별 자동 수정 예시
| 라벨 | 판정 조건 예시 | 자동 수정 방향 |
|---|---|---|
| `early_fall` | `success=false` and `fall_time_sec < 2.0` | orientation/collapse/height penalty 증가, target speed 감소, startup hold/ramp-up 증가 |
| `unstable` | `abs(mean_roll)>0.30` or `abs(mean_pitch)>0.30` | 자세 안정 패널티 증가, 램프 완화 |
| `standstill` | `success=true` but `mean_vel_error` 높음 | velocity tracking reward 강화, standstill penalty 강화 |
| `wrong_gait` | `contact_match_score < 0.65` | gait/contact 관련 항 가중치 및 phase/cadence 목표 보정 |
| `high_energy` | `energy` 상위 위험 구간 | energy/joint-vel penalty 증가 |
| `compute_timeout` | timeout 또는 반복 오버런 | Nsample/Hsample 축소, 실시간 배율 완화 |

## 6.3 정책 적용 순서
1. 하드 실패 수정 (`early_fall`, `compute_timeout`) 우선
2. 안정화 (`unstable`) 적용
3. 그 다음 동작 유사도(`wrong_gait`) 개선
4. 마지막 효율(`high_energy`) 최적화

---

## 7) 파일별 구현 대상 (코드 접점)

### 기존 파일 수정
- `main_sus.py`
- `agent_gemini.py`
- `gen_reward_code.py`
- `dial-mpc/dial_mpc/envs/unitree_go2_env.py`
- `dial-mpc/dial_mpc/envs/sds_reward_function.py`
- `dial-mpc/dial_mpc/examples/sds_gallop_sim.yaml`
- `dial-mpc/dial_mpc/core/dial_core.py` (배치 모드 옵션 필요 시)

### 신규 파일 추가 (automation 폴더 권장)
- `automation/run_single_experiment.py`
- `automation/run_sweep.py`
- `automation/classify_failure.py`
- `automation/propose_next_params.py`
- `automation/analyze_results.py`
- `automation/sweep_space.yaml`
- `automation/policy_rules.yaml`

### 7.1 스크립트별 I/O 계약
#### `automation/run_single_experiment.py`
- 입력
- `--config` YAML 경로
- `--outdir` trial 저장 경로
- `--seed` 정수 시드
- `--override key=value` 반복 입력 또는 `--reward.*` 직접 입력
- 출력
- 표준 산출물 5종 + 실패 시 최소 `metrics.json`

#### `automation/classify_failure.py`
- 입력
- `metrics.json`, `stdout.log`, `stderr.log`
- 출력
- `failure_label.json` (`label`, `reason`, `confidence`)

#### `automation/propose_next_params.py`
- 입력
- 최근 results CSV, `policy_rules.yaml`, `sweep_space.yaml`
- 출력
- 다음 trial 파라미터 (`proposed_params.yaml`)

#### `automation/run_sweep.py`
- 입력
- base config, 탐색 공간, 운영 옵션(trials/topk/timeout/resume)
- 출력
- trial 디렉터리, `results.csv`, `_best/`, sweep 로그

#### `automation/analyze_results.py`
- 입력
- `results.csv`
- 출력
- `summary.md`, `top10.csv`, `correlation.csv`

---

## 8) 실행 커맨드 표준안

## 8.1 단일 실험
```bash
python automation/run_single_experiment.py \
  --config dial-mpc/dial_mpc/examples/sds_gallop_sim.yaml \
  --outdir runs/single_2026-02-20/exp_000001 \
  --seed 123 \
  --reward.reward_vel_x_w 1.8 \
  --reward.penalty_collapse_w 10.0
```

## 8.2 스윕 실행
```bash
python automation/run_sweep.py \
  --config dial-mpc/dial_mpc/examples/sds_gallop_sim.yaml \
  --space automation/sweep_space.yaml \
  --out runs/sweep_2026-02-21 \
  --trials 500 \
  --topk 20 \
  --timeout-sec 1800 \
  --resume
```

## 8.3 야간 자동 실행 (서버)
```bash
0 2 * * * cd /home/<user>/SDS_DialMPC && \
  /home/<user>/miniconda3/envs/<env>/bin/python automation/run_sweep.py \
  --config dial-mpc/dial_mpc/examples/sds_gallop_sim.yaml \
  --space automation/sweep_space.yaml \
  --out runs/sweep_$(date +\%F) --trials 500 --topk 20 --timeout-sec 1800 --resume \
  >> logs/sweep_$(date +\%F).log 2>&1
```

## 8.4 결과 회수 (서버 -> 로컬)
```bash
rsync -azP <server>:/home/<user>/SDS_DialMPC/runs/sweep_2026-02-21 ./runs/
rsync -azP <server>:/home/<user>/SDS_DialMPC/logs/sweep_2026-02-21.log ./logs/
```

## 8.5 초기 smoke test 권장
```bash
python automation/run_sweep.py \
  --config dial-mpc/dial_mpc/examples/sds_gallop_sim.yaml \
  --space automation/sweep_space.yaml \
  --out runs/sweep_smoke \
  --trials 5 \
  --topk 2 \
  --timeout-sec 300 \
  --resume
```

---

## 9) 점수 함수 설계 원칙

## 9.1 기본 방향
- 하드 실패는 매우 큰 페널티
- 안정성/추종/비용을 균형 있게 반영
- 영상 유사도(contact/gait)를 명시적으로 반영

## 9.2 예시
```text
score =
  + episode_return
  - 300 * fall_count
  - 80  * mean_vel_error
  - 20  * (abs(mean_roll) + abs(mean_pitch))
  - 0.2 * energy
  + 40  * contact_match_score
```

---

## 10) 리스크와 예방책

### 리스크
- viewer/GUI 의존 실행으로 서버 headless에서 블로킹
- JAX/JIT 타이밍 이슈로 초반 제어 스파이크
- 디스크 용량 급증
- 실패 로그 누락으로 원인 추적 불가

### 예방책
- 배치 모드에서 GUI 의존성 제거
- startup hold / ramp-up / clamp 유지
- topK 외 대용량 산출물 정리 옵션
- stdout/stderr/returncode 항상 보존

### 10.1 장애 대응 런북 (운영 시)
#### 상황 A: trial이 연속 timeout
- `Nsample`, `Hsample` 즉시 하향
- `timeout-sec` 상향은 마지막 수단으로만 사용
- GPU 메모리 및 동시 프로세스 점검

#### 상황 B: success 비율 급감
- 최근 코드/설정 변경 diff 확인
- `sweep_space.yaml` 범위 과확장 여부 확인
- topK 기준 설정으로 롤백 실행

#### 상황 C: 넘어짐은 줄었지만 영상 유사도가 낮음
- `contact_match_score` 가중치 상향
- gait phase/cadence 관련 보상 항 강화
- 속도 추종과 안정성 비율을 재조정

#### 상황 D: 디스크 용량 임계 도달
- topK 외 산출물 정리
- 오래된 sweep 디렉터리 압축/아카이브
- raw dump 옵션 비활성화

---

## 11) 단계별 검증 체크리스트

## 11.1 1단계 체크
- [ ] API 키 하드코딩 제거
- [ ] 절대경로 제거
- [ ] 로컬/서버 동일 커맨드 실행 가능

## 11.2 2단계 체크
- [ ] 1회 실행 표준 결과 파일 생성
- [ ] 실패 시에도 `metrics.json` 생성
- [ ] reward 파라미터 CLI/YAML override 동작

## 11.3 3단계 체크
- [ ] 50~100 trial 연속 실행
- [ ] `--resume` 무중복 복구
- [ ] topK 폴더 유지

## 11.4 4단계 체크
- [ ] nightly 자동 실행
- [ ] 아침 요약 리포트 자동 생성
- [ ] 실패율/성능 추세 확인 가능

---

## 12) 이번 계획의 최종 운영 형태

1. 영상 입력 -> SUS 생성 -> 리워드 코드/파라미터 생성  
2. 자동 실험 루프가 넘어짐/불일치를 지속 교정  
3. 서버에서 야간 대량 탐색  
4. 아침 리포트 기반으로 탐색 범위 업데이트  
5. 베스트 설정만 보관해 실전 적용

이 문서 기준으로 구현하면, 현재 병목인 `넘어짐`과 `영상 불일치`를 수동 디버깅 중심에서 자동 튜닝 중심으로 전환할 수 있다.
