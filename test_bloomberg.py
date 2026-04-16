import subprocess
import os

url = "http://www.youtube.com/watch?v=iEpJwprxDdk" # Bloomberg
duration = 5
temp_path = "test_bloomberg.mp4"

print("Getting stream URL...")
stream_url = subprocess.check_output(['yt-dlp', '-g', '-f', 'best[height<=360]', url]).decode().strip()

print("Running ffmpeg...")
cmd = [
    'ffmpeg', '-y',
    '-i', stream_url,
    '-t', str(duration),
    '-c', 'copy',
    '-movflags', '+faststart',
    temp_path
]

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate(timeout=20)

if os.path.exists(temp_path):
    size = os.path.getsize(temp_path)
    print(f"Success! File size: {size} bytes")
    # Verify it opens with cv2
    import cv2
    cap = cv2.VideoCapture(temp_path)
    ret, frame = cap.read()
    if ret:
        print(f"Successfully read first frame: {frame.shape}")
    else:
        print("Failed to read frame")
else:
    print("Failed to create file.")
