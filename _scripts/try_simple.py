from pathlib import Path

from colorlog import exception

from video_reader import AsyncVideoReader

path_videos = Path("/home/kushal/data/gerbils/")
if not path_videos.exists():
    path_videos = Path("/Users/ebalzani/Code/pynaviz/tests/test_video")
    paths = [
        p for ext in ["avi", "mkv", "mp4"] for p in path_videos.glob("*.{}".format(ext))
    ]
else:
    paths = sorted(path_videos.glob("*.mp4"))

video = AsyncVideoReader(paths[0].as_posix())

frames = video[(slice(0,1), slice(0, 50))]



