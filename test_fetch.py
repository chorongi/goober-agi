import subprocess
import time
import os

url = "http://www.youtube.com/watch?v=m3kR2KK8TEs" # NASA Live
duration = 10
temp_path = "test_nasa.mp4"

print("Getting stream URL...")
stream_url = subprocess.check_output(['yt-dlp', '-g', '-f', 'best[height<=360]', url]).decode().strip()

print(f"Stream URL: {stream_url[:50]}...")
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
stdout, stderr = process.communicate(timeout=25)

if os.path.exists(temp_path):
    size = os.path.getsize(temp_path)
    print(f"Success! File size: {size} bytes")
else:
    print("Failed to create file.")
    print(stderr.decode()[-500:])
