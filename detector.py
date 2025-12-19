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

def process_video_peak_fixed(v_path, t_path, start_opt):
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

    is_tracking_peak = False
    peak_score = -1.0
    peak_frame_info = {} 
    last_record_time_sec = -1.0
    program_start_time = time.time()

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 분석 시작 (피크 탐색 및 추세 분석)...")

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
        runtime_ts = str(timedelta(seconds=int(time.time() - program_start_time)))
        
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

            # 피크 추적 로직
            if max_val >= params['threshold']:
                if not is_tracking_peak:
                    is_tracking_peak = True
                    peak_score = max_val
                    peak_frame_info = {'idx': frame_idx, 'vts': video_ts, 'score': max_val}
                else:
                    if max_val >= peak_score:
                        peak_score = max_val
                        peak_frame_info = {'idx': frame_idx, 'vts': video_ts, 'score': max_val}
                
                status_text = f"MATCHED! ({max_val:.2f}) .."
                box_color = (0, 0, 255) # 상승 중엔 빨간색
            
            # 값이 꺾였을 때 기록 (추적 중이고 현재값이 이전 최고점보다 낮거나 임계값 미만일 때)
            if is_tracking_peak and (max_val < peak_score or max_val < params['threshold']):
                p_idx = peak_frame_info['idx']
                p_vts = peak_frame_info['vts']
                p_score = peak_frame_info['score']
                p_sec = p_idx / fps # 오류 수정: peak_frame_info['vts_sec'] 대신 직접 계산
                
                elapsed = 0.0 if last_record_time_sec < 0 else (p_sec - last_record_time_sec)
                
                with open(params['output_csv'], mode='a', newline='', encoding='utf-8-sig') as f:
                    csv.writer(f).writerow([p_idx, p_vts, f"{elapsed:.3f}", round(p_score, 4)])
                
                print(f">>> [Runtime {runtime_ts}] [RECORDED PEAK] Video: {p_vts} | Score: {p_score:.4f} | Elapsed: {elapsed:.3f}s")
                
                last_record_time_sec = p_sec
                is_tracking_peak = False
                peak_score = -1.0
                
                # 쿨타임 점프
                if params['cooldown_sec'] > 0:
                    jump_frame = int(frame_idx + (params['cooldown_sec'] * fps))
                    if jump_frame < total_frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, jump_frame)
                        frame_idx = jump_frame
                        continue
            
            if not is_tracking_peak:
                status_text = f"Searching... ({max_val:.2f})"
                box_color = (0, 255, 255)
        else:
            status_text = "SKIP (Cooldown)"
            box_color = (100, 100, 100)

        if params['show_display']:
            display_frame = frame_resized.copy()
            cv2.putText(display_frame, f"Runtime: {runtime_ts}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Video: {video_ts}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if is_cooldown_over:
                cv2.rectangle(display_frame, top_left, (top_left[0]+tw, top_left[1]+th), box_color, 2)
                cv2.putText(display_frame, status_text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            else:
                cv2.putText(display_frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 1)

            cv2.imshow("Analysis", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 분석이 완료되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", required=True)
    parser.add_argument("-t", "--template", required=True)
    parser.add_argument("-s", "--start", type=float)
    args = parser.parse_args()
    process_video_peak_fixed(args.video, args.template, args.start)