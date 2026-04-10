from __future__ import annotations

import os

# decord uses up all the RAM otherwise
os.environ["DECORD_EOF_RETRY_MAX"] = "128"
# os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libcuda.so:/usr/lib/x86_64-linux-gnu/libnvcuvid.so:$LD_PRELOAD"

from concurrent.futures import Future
import multiprocessing
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import signal
import threading

import numpy as np
try:
    from decord import VideoReader
except ImportError:
    VideoReader = None

from . import VideoHandler, av

from _vr_process import _reader_process

mp_ctx = multiprocessing.get_context("spawn")


class AsyncVideoReader:
    def __array__(self) -> AsyncVideoReader:
        return self

    def __init__(self, path: str | Path, **kwargs):
        self._path = Path(path)
        self._kwargs = kwargs

        vr = VideoReader(str(self._path), num_threads=1)
        try:
            frame0 = vr[10].asnumpy()
            vr.seek(0)
        except IndexError:
            frame0 = vr[0].asnumpy()
            vr.seek(0)

        self._shape = (len(vr), *frame0.shape)
        self._dtype = np.dtype(frame0.dtype)
        del vr

        self._shm = SharedMemory(create=True, size=frame0.nbytes)

        self._request_queue: Queue = mp_ctx.Queue()
        self._response_queue: Queue = mp_ctx.Queue()

        self._pending_rid: int = 0
        self._pending_future: Future | None = None
        self._lock = threading.Lock()

        self._worker = mp_ctx.Process(
            target=_reader_process,
            kwargs=dict(
                path=self._path,
                kwargs=self._kwargs,
                shm_name=self._shm.name,
                frame_shape=(1, *frame0.shape),
                dtype=str(self._dtype),
                request_queue=self._request_queue,
                response_queue=self._response_queue,
            ),
            daemon=True,
        )
        self._worker.start()

        self._listener = threading.Thread(target=self._listen, daemon=True)
        self._listener.start()

        self._result = np.ndarray((1, *frame0.shape), dtype=self.dtype, buffer=self._shm.buf)

    def _listen(self):
        while True:
            msg = self._response_queue.get()
            if msg is None:
                break

            rid, frame_shape, dtype, shm_name = msg

            with self._lock:
                if rid != self._pending_rid:
                    continue

                if shm_name != self._shm.name:
                    self._shm.unlink()
                    self._shm.close()
                    self._shm = SharedMemory(name=shm_name)

                # TODO: when buffer size changes
                # result = np.ndarray(frame_shape, dtype=dtype, buffer=self._shm.buf)
                future = self._pending_future

            future.set_result(self._result)

    def __getitem__(self, index) -> Future:
        with self._lock:
            # if a new frame has been requested before the previous frame finished decoding
            # TODO: make this toggleable, use this only when the slider is moving around
            #  else if the user has clicked the "play" or "step" button we want to render EVERY frame!!!!
            if self._pending_future is not None and not self._pending_future.done():
                self._pending_future.cancel()
                print("cancelled")
                # tell it to stop decoding/copying the current frame and start the next frame
                os.kill(self._worker.pid, signal.SIGUSR1)

            self._pending_rid += 1
            # a trick to get the resultant shape of the sliced array with a zero-memory dummy array
            final_shape = np.empty(shape=self.shape, dtype="V0")[index[0]].shape
            future = Future()
            self._pending_future = future

        self._request_queue.put((self._pending_rid, index[0]))
        return future

    def shutdown(self, wait: bool = True):
        self._request_queue.put(None)
        if wait:
            self._worker.join()
            self._response_queue.put(None)
            self._listener.join()
        self._shm.unlink()
        self._shm.close()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)


if __name__ == "__main__":
    import fastplotlib as fpl
    import pyinstrument
    paths = sorted(Path("/home/kushal/data/gerbils/").glob("*.mp4"))

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
