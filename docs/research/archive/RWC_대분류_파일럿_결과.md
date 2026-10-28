# RWC-G 소규모 대분류 평가

> 과거 실험 기록 · 공개본 2026-10-29. 본문의 모델·라벨·기본값·다음 단계는 작성 당시의 상태입니다. 현재 0.6.0 familiar 구성의 정확도를 뜻하지 않습니다. [현재 구성과 해석 범위](../README.md)를 먼저 확인하세요. 본문의 로컬 원시 자료·실험 도구는 이 공개본에 포함하지 않습니다.


2026-10-26 · 새 공식 WAV18항목 · API 0.5.0 유지

## 결과

RWC-G의 제작 장르를 기준으로 고정한 18항목을 실제 확보하고 세 모델을 비교했다. 이 결과는 **제작 카탈로그 대분류와의 일치도**다. 현대 J-pop·자체 9대분류·세부 장르 정확도를 검증한 결과로 해석하지 않는다.

| 모델 | 첫 후보의 제작 대분류 일치 | 상위3개 중 제작 대분류 포함 |
|---|---:|---:|
| 기존 Wav2Vec2/GTZAN | 10/18 (55.6%) | 14/18 |
| CLAP unfused | 7/18 (38.9%) | 14/18 |
| MuQ-MuLan | 17/18 (94.4%) | 18/18 |

같은 범주만 항상 내는 단순 기준은 2/18이다. 실제 제작 스타일이 복합적일 수 있고, 한 개의 제작 라벨이 청취자가 느끼는 모든 장르를 포괄하지는 않는다. 이 수치를 새 음악의 정답 확률이나 일반적인 모델 정확도로 표시하지 않는다.

각 참조 범주가2항목씩이라 대표된9범주의 평균 재현율은 이번 표의 첫 후보 일치율과 같다. 예측 후보는 원래10장르 전체를 유지했다. disco 양성 예제는 없어 disco 검증을 완료했다고 할 수 없다. 18개의 서로 다른 ID 기반 곡 그룹이지만 아티스트 이름은12개이며 반복이 있다. 독립18아티스트로 간주하는 유의성 검정·부트스트랩은 하지 않았다.

## 범주별 첫 후보 일치 수

| 제작 참조 범주 | 곡 수 | 기존 모델 | CLAP | MuQ |
|---|---:|---:|---:|---:|
| pop | 2 | 0 | 0 | 2 |
| rock | 2 | 2 | 2 | 2 |
| metal | 2 | 1 | 1 | 1 |
| hiphop | 2 | 1 | 1 | 2 |
| jazz | 2 | 2 | 2 | 2 |
| reggae | 2 | 1 | 0 | 2 |
| classical | 2 | 2 | 1 | 2 |
| blues | 2 | 0 | 0 | 2 |
| country | 2 | 1 | 0 | 2 |

## 대상 선정과 원본 확인

- 이전 조사에서 현재10장르와 보수적으로 대응한 RWC-G 후보 중100MiB 이하인 항목만 사용했다. 9개 참조 범주별2항목을 seed `rwc-genre-pilot-20261026-v1`로 선정했고, 작품 ID/악장 중복을 피하며 이미 사용한 아티스트 이름의 재사용을 줄이는 순서를 적용했다.
- 선정18개와 라벨 대응은 음원 다운로드·모델 점수 확인 전에 고정했다. 확보 실패를 성능이 좋은 곡으로 바꾸지 않았다. J-pop80곡을 pop 정답으로 넣지 않았다.
- 공식 RWC-G.zip의 필요한 항목만 HTTP Range로 받았다. 전체 약3.9GB ZIP은 받지 않았다. 선택한 압축 항목 크기 합은 약730.3MB다. 이어받기 과정에 재시도가 있었으므로 이 합계를 실제 네트워크 총사용량으로 단정하지 않는다.
- ZIP 로컬 헤더·파일명·크기·중앙 디렉터리 CRC32를 확인했고 원본 WAV 바이트의 로컬 SHA256을 기록했다. 전체 ZIP의 게시 MD5를 검증한 것은 아니며 로컬 SHA를 게시자 제공 음원 SHA라고 표시하지 않는다.
- 실제 디코딩의 유한 값·비무음·최소 길이·메타데이터 길이를 확인했다. 음원을 자르거나 정규화하거나 다른 포맷으로 저장하지 않았다. 모든 입력은 기존100MiB 제한 이하다.
- 전송이 느려 이미 검증한 파일을 재사용하고 병렬도를2에서4로 높여 이어받았다. 전환 중 겹친 실행은 정리하고 저장된11개 원본을 다시 CRC·해시로 확인했다. 최종 평가는18개 모두 확보된 뒤 시작했다. 곡/참조/평가 기준은 바꾸지 않았으며 전환 기록을 보관했다.

RWC는 공개 연구용으로 재배포된 기존 제작 녹음이다. 배포·귀속 조건은 [공식 오디오 레코드](https://zenodo.org/records/18656623), 제작 장르의 정의와 편의상 분류라는 설명은 [원 장르 문서](https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-g.html)를 따른다. 원 저자·데이터 출처: Masataka Goto 외, RWC Music Database, 장르 DB 논문 ISMIR2003 및 공개 재배포 v2(2026).

## 같은 분석 조건

- 기존 비교의 모델 revision·문구·점수 평균을 그대로 사용했다. Wav2Vec2·CLAP·MuQ 모두 같은 원본 오디오 시간 구간을 분석했다.
- 전체 입력을 모노48kHz 기준으로 맞춘 뒤 균등한 최대10초 구간으로 나누고 꼬리를 유지했다. 각 백엔드는 자기 표본률·기존 패딩 규칙을 사용한다. 장르 점수는 구간별로 평균했다.
- 이는 현재 API의 기본 Wav2Vec2 첫 최대30초 한 번 분류와 구분되는 실험 처리다. 원시 점수는 모델 사이에 크기를 비교하거나 합치지 않았다.
- 모든 모델·음원을 성공 처리한 고정18항목에서만 비교했다. 기본 모델·프롬프트·임계값을 조정하거나 새 학습을 수행하지 않았다.

## 검증 범위와 한계

선정 항목에는 메타데이터상 일본어 가창으로 표시된 곡이 없다. RWC-G의 일반 대분류 참조를 시험한 것이며, J-pop 내부 스타일 구분을 입증하는 자료가 아니다. 오래된 제작 녹음이고 모델 사전학습과의 중복은 확인되지 않았다. 이전 MTG-Jamendo의 불완전한 복수 업로더 태그 평가와도 참조 의미가 다르므로 두 결과를 합산하지 않는다.

전체 테스트217개가 통과했다. 별도 감사에서 고정 원본의 CRC·로컬SHA, 동일 구간, 구간 점수 평균, 참조 대응과 모든 일치도 지표를 다시 계산했다. 통계적 우열이나 모든 장르의 우수성을 확정하지 않는다.

| 모델 | 적재·실행 포함 처리시간 |
|---|---:|
| 기존 Wav2Vec2/GTZAN | 34.4초 |
| CLAP unfused | 54.0초 |
| MuQ-MuLan | 68.4초 |

Windows·RTX4070에서 순차 실행한 단일 측정이다. 캐시와 시작 비용을 교차 통제한 속도 실험은 아니다.

## 항목별 첫 후보

| ID | 제목 | 원래 대분류 / 하위 분류 | 평가 참조 | 기존 모델 | CLAP | MuQ |
|---|---|---|---|---|---|---|
| RWC_G002 | Forget about It | Popular / Popular | pop | rock | rock | pop |
| RWC_G003 | In Your Arms | Popular / Popular | pop | country | rock | pop |
| RWC_G008 | Living in a World of Destruction | Rock / Rock | rock | rock | rock | rock |
| RWC_G009 | Waiting for Your Love | Rock / Rock | rock | rock | rock | rock |
| RWC_G010 | Lost in My Dreams | Rock / Heavy Metal | metal | metal | metal | metal |
| RWC_G012 | Woman | Rock / Heavy Metal | metal | pop | rock | rock |
| RWC_G014 | Tuesday Morning | Dance / Rap / Hip-Hop | hiphop | hiphop | hiphop | hiphop |
| RWC_G015 | Rocket in My Pocket | Dance / Rap / Hip-Hop | hiphop | pop | pop | hiphop |
| RWC_G030 | Kitchen | Jazz / Big Band | jazz | jazz | jazz | jazz |
| RWC_G031 | Azure | Jazz / Modern Jazz | jazz | jazz | jazz | jazz |
| RWC_G044 | Eric's Reggae | Latin / Reggae | reggae | disco | rock | reggae |
| RWC_G045 | Where Are You? | Latin / Reggae | reggae | reggae | blues | reggae |
| RWC_G057 | Toccata and Fugue in D minor, BWV 565 | Classical / Baroque | classical | classical | jazz | classical |
| RWC_G060 | String Quartet No.77 in C major, Op.76 No.3, Hob.III:77, "Kaiser", 2nd mvmt. | Classical / Classical | classical | classical | classical | classical |
| RWC_G065 | Got'em Both | World / Blues | blues | rock | rock | blues |
| RWC_G066 | Dear John's Letter | World / Blues | blues | rock | rock | blues |
| RWC_G071 | Desperate Little Man | World / Country | country | country | blues | country |
| RWC_G072 | I Don't Love Nobody | World / Country | country | blues | blues | country |

## 보관 위치

`data/evaluation/rwc_genre_pilot_v1/`에 선정·확보 명세, `data/evaluation/rwc_genre_evaluation_v1/`에 고정 실행 계획·모델별 전체 점수·요약·감사 결과를 보관했다. 원본 오디오는 `work/rwc-genre-pilot/audio/`에 있다. 근거 감사는 `scripts/audit_rwc_genre_pilot.py`로 실행할 수 있다.

현재 사용자가 직접 장르를 판정하거나 라벨을 작성할 필요는 없다. 연구 범위에 맞는 제작 참조를 사용했고, 범위를 벗어나는 J-pop·자체9장르·세부 장르의 검증은 아직 별도로 남아 있다.
