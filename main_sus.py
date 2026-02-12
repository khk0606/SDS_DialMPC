import cv2
import numpy as np
import base64
import os
from yt_dlp import YoutubeDL
from agent_gemini import SUSGenerator

# 설정
cfg = {
    "model": "gemini-2.5-flash" 
}
OUTPUT_DIR = "output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def download_video(url, output_path="input_video.mp4"):
    print(f"Downloading video from {url}...")
    ydl_opts = {
        'format': 'best[ext=mp4]', 
        'outtmpl': output_path, 
        'quiet': True,
        'overwrites': True
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Download complete: {output_path}")
    return output_path

def create_frame_grid(video_path, grid_size=(2, 4)):
    print("Extracting frames...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 영상 전체에서 균일하게 8장 추출
    indices = np.linspace(0, total_frames-10, grid_size[0]*grid_size[1], dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # AI 처리를 위해 적당한 크기로 리사이즈 (320x240)
            frame = cv2.resize(frame, (320, 240)) 
            frames.append(frame)
    cap.release()
    
    # 그리드 이미지 생성
    rows = []
    for r in range(grid_size[0]):
        row_frames = frames[r*grid_size[1] : (r+1)*grid_size[1]]
        if row_frames:
            rows.append(np.hstack(row_frames))
            
    if not rows:
        raise ValueError("No frames extracted!")
        
    grid_image = np.vstack(rows)
    
    # 확인용 저장 (WSL은 창을 못 띄우므로 파일로 확인)
    save_path = os.path.join(OUTPUT_DIR, "frame_grid_preview.png")
    cv2.imwrite(save_path, grid_image)
    print(f"Frame grid saved to: {save_path}")
    
    return grid_image

def encode_image_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

# === 실행 ===
if __name__ == "__main__":
    # 1. Unitree Go2에 적합한 영상 (개/사족보행 로봇)
    video_url = "https://www.youtube.com/watch?v=A8YPHoc0dR0" 
    
    try:
        # 비디오 준비
        video_path = os.path.join(OUTPUT_DIR, "target_motion.mp4")
        download_video(video_url, video_path)
        
        # 그리드 이미지 생성
        grid_img = create_frame_grid(video_path)
        encoded_grid = encode_image_base64(grid_img)
        
        # SUS 파이프라인 실행
        print("\n>>> Starting SUS Analysis Pipeline <<<")
        sus_gen = SUSGenerator(cfg)
        final_report = sus_gen.generate_sus_prompt(encoded_grid)
        
        # 결과 저장
        report_path = os.path.join(OUTPUT_DIR, "final_sus_report.txt")
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(final_report)
            
        print("\n" + "="*50)
        print(f" Analysis Complete! Report saved to: {report_path}")
        print("="*50)
        print(final_report)
        
    except Exception as e:
        print(f"\n Error occurred: {e}")