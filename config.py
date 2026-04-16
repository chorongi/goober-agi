# Benchmark Configuration

VIDEO_SOURCES = {
    "Music & Study": [
        {"name": "Lofi Girl", "url": "https://www.youtube.com/watch?v=jfKfPfyJRdk"},
        {
            "name": "Monstercat Silk",
            "url": "http://www.youtube.com/watch?v=WsDyRAPFBC8",
        },
        {"name": "Chillhop Music", "url": "http://www.youtube.com/watch?v=7NOSDKb0HlU"},
    ],
    "Nature & Animals": [
        {
            "name": "Explore.org Puppy Cam",
            "url": "http://www.youtube.com/watch?v=h-Z0wCdD3dI",
        },
        {"name": "iPanda", "url": "http://www.youtube.com/watch?v=9LvjI3NelAU"},
        {"name": "NamibiaCam", "url": "https://www.youtube.com/watch?v=ydYDqZQpim8"},
        {"name": "AfriCam", "url": "https://www.youtube.com/watch?v=qpukdDslCjk"},
        {
            "name": "Wildlife In The Forest",
            "url": "https://www.youtube.com/watch?v=F0GOOP82094",
        },
    ],
    "News & Finance": [
        {"name": "Sky News", "url": "http://www.youtube.com/watch?v=YDvsBbKfLPA"},
        {
            "name": "Al Jazeera English",
            "url": "http://www.youtube.com/watch?v=YDvsBbKfLPA",
        },
        {
            "name": "Bloomberg Television",
            "url": "http://www.youtube.com/watch?v=iEpJwprxDdk",
        },
    ],
    "Space & Science": [
        {"name": "NASA Live", "url": "http://www.youtube.com/watch?v=m3kR2KK8TEs"},
        {"name": "Blooket Live", "url": "http://www.youtube.com/watch?v=M5OKNwczOP8"},
    ],
    "Urban & Transprot": [
        {"name": "La Plata", "url": "https://www.youtube.com/watch?v=X-ir2KfXMX0"},
        {"name": "Tokyo Streets", "url": "https://www.youtube.com/watch?v=L6wO1-U2RTY"},
    ],
}

# Task Parameters
TASK1_WINDOW_SEC = 10
TASK2_WINDOW_SEC = 10
TASK2_FPS = 1
TASK3_BASELINE_MIN = 5
TASK3_STABILITY_WINDOWS = 3

# Weights for final score (Total 100)
WEIGHTS = {"task1": 0.333, "task2": 0.333, "task3": 0.334}
