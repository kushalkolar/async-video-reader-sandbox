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
from decord import VideoReader
from _vr_process import _reader_process

mp_ctx = multiprocessing.get_context("spawn")


class AsyncVideoReader:
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
                frame_shape=frame0.shape,
                dtype=str(self._dtype),
                request_queue=self._request_queue,
                response_queue=self._response_queue,
            ),
            daemon=True,
        )
        self._worker.start()

        self._listener = threading.Thread(target=self._listen, daemon=True)
        self._listener.start()

        self._result = np.ndarray(frame0.shape, dtype=self.dtype, buffer=self._shm.buf)

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

    def __getitem__(self, index: int) -> Future:
        with self._lock:
            # if a new frame has been requested before the previous frame finished decoding
            # TODO: make this toggleable, use this only when the slider is moving around
            #  else if the user has clicked the "play" or "step" button we want to render EVERY frame!!!!
            if self._pending_future is not None and not self._pending_future.done():
                self._pending_future.cancel()
                # tell it to stop decoding/copying the current frame and start the next frame
                os.kill(self._worker.pid, signal.SIGUSR1)

            self._pending_rid += 1
            future = Future()
            self._pending_future = future

        self._request_queue.put((self._pending_rid, index))
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
    from fastplotlib.ui import EdgeWindow
    from functools import partial
    from imgui_bundle import imgui
    from time import perf_counter

    INDEX = 0
    INDEX_VIEW = 0

    paths = sorted(Path("/home/kushal/data/gerbils/").glob("*.mp4"))

    vrs = list()
    for p in paths:
        vrs.append(AsyncVideoReader(p))

    fig = fpl.Figure(shape=(2, 2), size=(1000, 1000))

    images = list()
    for subplot, vr in zip(fig, vrs):
        f = vr[0]
        data = f.result(5)
        images.append(subplot.add_image(data[::2, ::2]))

    def update_index(i):
        global images
        global INDEX_VIEW
        INDEX_VIEW = i

        futures = list()
        for vr, g in zip(vrs, images):
            future = vr[i]
            future.add_done_callback(partial(_update_graphic, g))
            futures.append(future)

        while all([f.running() for f in futures]):
            pass

    def _update_graphic(g, fut: Future):
        if fut.cancelled():
            return
        data = fut.result()
        g.data = data[::2, ::2]

    class ImageProcessingWindow(EdgeWindow):
        def __init__(self, figure, size, location, title):
            super().__init__(figure=figure, size=size, location=location, title=title)
            self._last_update = 0
            self._playing = False

        def update(self):
            global INDEX
            if not self._playing:
                if imgui.button("play"):
                    self._playing = True
            else:
                if imgui.button("pause"):
                    self._playing = False

            if self._playing:
                if perf_counter() - self._last_update > (1 / 50):
                    INDEX += 1
                    update_index(INDEX)
                    self._last_update = perf_counter()

            changed, value = imgui.slider_int(label="sigma", v=INDEX, v_min=0, v_max=vrs[0].shape[0])
            if changed:
                if perf_counter() - self._last_update > 0.1:
                    update_index(value)
                    self._last_update = perf_counter()
                INDEX = value

            elif perf_counter() - self._last_update > 0.5:
                if INDEX_VIEW != INDEX:
                    update_index(INDEX)

    gui = ImageProcessingWindow(fig, size=200, location="bottom", title="bah")

    fig.add_gui(gui)


    fig.show()
    fpl.loop.run()
