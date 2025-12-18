import cv2
import csv
import os
import configparser
from datetime import datetime

def load_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    if not os.path.exists(config_file): return None
    config.read(config_file, encoding='utf-8')
    s = config['SETTINGS']
    return {
        'video_path': s.get('video_path'),
        'template_path': s.get('template_path'),
        'threshold': s.getfloat('threshold'),
        'output_csv': s.get('output_csv'),
        'show_display': s.getboolean('show_display'),
        'min_frames': s.getint('min_consecutive_frames'),
        'cooldown_sec': s.getfloat('record_cooldown_seconds'),
        'skip_frames': s.getint('skip_frames', fallback=0)
    }

def process_video_clean_log():
    params = load_config()
    if not params:
        print("설정 파일을 확인하세요.")
        return

    cap = cv2.VideoCapture(params['video_path'])
    template_orig = cv2.imread(params['template_path'], cv2.IMREAD_COLOR)
    if not cap.isOpened() or template_orig is None:
        print("파일을 열 수 없습니다.")
        return

    # 50% 축소 처리
    template = cv2.resize(template_orig, (0, 0), fx=0.25, fy=0.25)
    th, tw = template.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, first_frame = cap.read()
    first_frame_resized = cv2.resize(first_frame, (0, 0), fx=0.25, fy=0.25)
    roi = cv2.selectROI("Select ROI", first_frame_resized, False)
    x, y, w, h = roi
    use_roi = True if w > 0 and h > 0 else False
    cv2.destroyWindow("Select ROI")

    # CSV 초기화
    with open(params['output_csv'], mode='w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(['Frame', 'Video_Timestamp', 'Score'])

    consecutive_count = 0
    is_detecting = False
    last_record_time = -9999.0
    frame_idx = 0

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 분석을 시작합니다...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        if params['skip_frames'] > 0 and frame_idx % (params['skip_frames'] + 1) != 0:
            frame_idx += 1
            continue

        frame_resized = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        search_area = frame_resized[y:y+h, x:x+w] if use_roi else frame_resized
        res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        current_time_sec = frame_idx / fps
        video_ts = f"{int(current_time_sec // 60):02d}:{current_time_sec % 60:06.3f}"
        
        is_cooldown_over = (current_time_sec - last_record_time) >= params['cooldown_sec']

        # --- 감지 및 기록 로직 ---
        if max_val >= params['threshold']:
            consecutive_count += 1
            if not is_detecting and is_cooldown_over:
                # CSV 기록
                with open(params['output_csv'], mode='a', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow([frame_idx, video_ts, round(max_val, 4)])
                
                is_detecting = True
                last_record_time = current_time_sec
                
                # 콘솔 로그 (발견 시에만 출력)
                log_now = datetime.now().strftime('%H:%M:%S')
                print(f">>> [{log_now}] [RECORDED] Video Time: {video_ts} | Score: {max_val:.4f}")
        else:
            if consecutive_count >= params['min_frames']:
                is_detecting = False
            consecutive_count = 0

        # --- 화면 표시 (GUI) ---
        if params['show_display']:
            display_frame = frame_resized.copy()
            top_left = (max_loc[0] + x, max_loc[1] + y) if use_roi else max_loc
            bottom_right = (top_left[0] + tw, top_left[1] + th)
            
            # 색상 및 텍스트 설정
            if max_val >= params['threshold']:
                box_color, status_text = (0, 0, 255), f"MATCHED! ({max_val:.2f})"
            else:
                box_color, status_text = (0, 255, 255), f"Searching... ({max_val:.2f})"

            cv2.rectangle(display_frame, top_left, bottom_right, box_color, 2)
            cv2.putText(display_frame, status_text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            cv2.putText(display_frame, f"Video: {video_ts}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Template Matching Analysis", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 분석이 완료되었습니다.")

if __name__ == "__main__":
    process_video_clean_log()