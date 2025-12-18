# MatchingTimeChecker
템플릿 이미지와 매칭되는 지점의 동영상 시간을 체크하는 프로그램

# Installation
## OpenCV, Pandas, NumPy 설치
    
    pip install opencv-python pandas numpy

# Run
    
    python ./detector.py -v ./video.mp4 -t ./template.png
    
# configs
config.ini 파일 수정

- 일치 판단 임계값 (0.0 ~ 1.0)
  - threshold = 0.75
- 결과 저장 파일명
  - output_csv = result.csv
- 화면 표시 여부 (True / False)
  - show_display = False
- 최소 연속 프레임 간격 
  - min_consecutive_frames = 1
- 기록 후 재기록까지 대기 시간 (초 단위)
    - 예: 5로 설정하면 기록 후 5초 동안은 다시 기록하지 않음
    - record_cooldown_seconds = 25
- 분석 시 이미지 축소 비율 (1.0 = 원본, 0.5 = 50%, 0.25 = 25%)
    - resize_ratio = 0.5
- 0이면 모든 프레임 검사, 1이면 한 프레임씩 건너뜀 (속도 2배)
    - skip_frames = 1
- 시작 시 건너뛸 시간 (초)
    - start_second = 0