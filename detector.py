import cv2
import csv
import os
import configparser
import argparse
import time
from datetime import datetime, timedelta

def load_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    defaults = {
        'threshold': 0.8, 'output_csv': 'result.csv', 'show_display': True,
        'min_frames': 5, 'cooldown_sec': 5.0, 'resize_ratio': 0.5, 'skip_frames': 0,
        'start_second': 0.0
    }
    if not os.path.exists(config_file): return defaults
    config.read(config_file, encoding='utf-8')
    s = config['SETTINGS'] if 'SETTINGS' in config else {}
    return {
        'threshold': s.getfloat('threshold', fallback=0.8),
        'output_csv': s.get('output_csv', fallback='result.csv'),
        'show_display': s.getboolean('show_display', fallback=True),
        'min_frames': s.getint('min_consecutive_frames', fallback=5),
        'cooldown_sec': s.getfloat('record_cooldown_seconds', fallback=5.0),
        'resize_ratio': s.getfloat('resize_ratio', fallback=0.5),
        'skip_frames': s.getint('skip_frames', fallback=0),
        'start_second': s.getfloat('start_second', fallback=0.0)
    }

def process_video_final(v_path, t_path, start_opt):
    params = load_config()
    start_sec = start_opt if start_opt is not None else params['start_second']
    
    cap = cv2.VideoCapture(v_path)
    template_orig = cv2.imread(t_path, cv2.IMREAD_COLOR)
    
    if not cap.isOpened() or template_orig is None:
        print("파일 경로를 확인해주세요."); return

    ratio = params['resize_ratio']
    template = cv2.resize(template_orig, (0, 0), fx=ratio, fy=ratio)
    th, tw = template.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 시작 지점 점프
    start_frame = int(start_sec * fps)
    if 0 < start_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
    else:
        frame_idx = 0

    # ROI 선택
    cap.grab()
    ret, first_frame = cap.retrieve()
    if not ret: return
    first_frame_resized = cv2.resize(first_frame, (0, 0), fx=ratio, fy=ratio)
    roi = cv2.selectROI("Select ROI", first_frame_resized, False)
    x, y, w, h = roi
    use_roi = True if w > 0 and h > 0 else False
    cv2.destroyWindow("Select ROI")

    with open(params['output_csv'], mode='w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(['Frame', 'Video_Timestamp', 'Elapsed_Time', 'Score'])

    consecutive_count = 0
    is_detecting = False
    last_record_time_sec = -1.0
    
    # 프로그램 시작 시간 기록
    program_start_time = time.time()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 분석 시작...")

    while frame_idx < total_frames:
        if params['skip_frames'] > 0:
            for _ in range(params['skip_frames']):
                cap.grab()
                frame_idx += 1
            if frame_idx >= total_frames: break

        ret, frame = cap.read()
        if not ret: break
        
        current_time_sec = frame_idx / fps
        video_ts = f"{int(current_time_sec // 60):02d}:{current_time_sec % 60:06.3f}"
        
        # 프로그램 가동 시간 계산 (Runtime)
        runtime_sec = int(time.time() - program_start_time)
        runtime_ts = str(timedelta(seconds=runtime_sec))
        
        is_cooldown_over = last_record_time_sec < 0 or (current_time_sec - last_record_time_sec) >= params['cooldown_sec']
        frame_resized = cv2.resize(frame, (0, 0), fx=ratio, fy=ratio)

        status_text = ""
        box_color = (150, 150, 150)
        top_left = (0, 0)

        if is_cooldown_over:
            search_area = frame_resized[y:y+h, x:x+w] if use_roi else frame_resized
            res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            top_left = (max_loc[0] + x, max_loc[1] + y) if use_roi else max_loc

            if max_val >= params['threshold']:
                consecutive_count += 1
                status_text = f"MATCHED! ({max_val:.2f})"
                box_color = (0, 0, 255) # 빨간색

                if not is_detecting:
                    elapsed = 0.0 if last_record_time_sec < 0 else (current_time_sec - last_record_time_sec)
                    with open(params['output_csv'], mode='a', newline='', encoding='utf-8-sig') as f:
                        csv.writer(f).writerow([frame_idx, video_ts, f"{elapsed:.3f}", round(max_val, 4)])
                    
                    is_detecting = True
                    last_record_time_sec = current_time_sec
                    print(f">>> [Runtime {runtime_ts}] [RECORDED] Video: {video_ts} | Score: {max_val:.4f}")

                    if params['cooldown_sec'] > 0:
                        jump_frame = int(frame_idx + (params['cooldown_sec'] * fps))
                        if jump_frame < total_frames:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, jump_frame)
                            frame_idx = jump_frame
                            is_detecting = False
                            consecutive_count = 0
                            continue
            else:
                if consecutive_count >= params['min_frames']: is_detecting = False
                consecutive_count = 0
                status_text = f"Searching... ({max_val:.2f})"
                box_color = (0, 255, 255) # 노란색
        else:
            status_text = "SKIP (Cooldown)"
            box_color = (100, 100, 100)

        # 화면 표시 (시각 효과 집중)
        if params['show_display']:
            display_frame = frame_resized.copy()
            
            # 1. 왼쪽 상단 정보 (프로그램 가동 시간 & 비디오 시간)
            cv2.putText(display_frame, f"Runtime: {runtime_ts}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Video: {video_ts}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 2. 매칭 상태 및 사각형 표시
            if is_cooldown_over:
                cv2.rectangle(display_frame, top_left, (top_left[0]+tw, top_left[1]+th), box_color, 2)
                cv2.putText(display_frame, status_text, (top_left[0], top_left[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            else:
                cv2.putText(display_frame, status_text, (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 1)

            cv2.imshow("Analysis", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 분석 종료 (총 가동 시간: {runtime_ts})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", required=True)
    parser.add_argument("-t", "--template", required=True)
    parser.add_argument("-s", "--start", type=float, help="시작 시간(초)")
    args = parser.parse_args()
    process_video_final(args.video, args.template, args.start)