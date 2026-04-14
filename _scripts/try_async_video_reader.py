import fastplotlib as fpl
import pyinstrument
from pathlib import Path
from video_reader import AsyncVideoReader

path_videos = Path("/home/kushal/data/gerbils/")
if not path_videos.exists():
    path_videos = Path("/Users/ebalzani/Code/pynaviz/tests/test_video")
    paths = [
        p for ext in ["avi", "mkv", "mp4"] for p in path_videos.glob("*.{}".format(ext))
    ]
else:
    paths = sorted(path_videos.glob("*.mp4"))


vrs = list()
for p in paths:
    vrs.append(AsyncVideoReader(p))

ref_ranges = {"t": (0, vrs[0].shape[0], 1)}
ndw = fpl.NDWidget(ref_ranges=ref_ranges, shape=(1, 4), size=(1800, 500))
for i, vr in enumerate(vrs):
    ndw[0, i].add_nd_image(vr, dims=list("tmnc"), spatial_dims=list("mnc"), rgb_dim="c", compute_histogram=False)

ndw.show()

run_profile = False

if run_profile:
    ndw.indices["t"] = 5000
    ndw._sliders_ui._playing["t"] = True

    with pyinstrument.Profiler(async_mode="enabled") as profiler:
        fpl.loop.run()

    profiler.print()
    profiler.open_in_browser()

else:
    fpl.loop.run()
