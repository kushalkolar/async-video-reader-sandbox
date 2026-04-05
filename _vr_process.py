import os

# decord uses up all the RAM otherwise
os.environ["DECORD_EOF_RETRY_MAX"] = "128"
# os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libcuda.so:/usr/lib/x86_64-linux-gnu/libnvcuvid.so:$LD_PRELOAD"

import multiprocessing
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import signal

import numpy as np

mp_ctx = multiprocessing.get_context("spawn")

interrupt = False

def _handle_sigusr1(signum, frame):
    global interrupt
    interrupt = True

# https://docs.python.org/3/library/signal.html#general-rules
# handles interrupt from main process
signal.signal(signal.SIGUSR1, _handle_sigusr1)


def _reader_process(path: Path, kwargs: dict, shm_name: str, frame_shape: tuple,
                    dtype: str, request_queue: Queue, response_queue: Queue):
    import os
    from decord import VideoReader, gpu, cpu

    os.environ["DECORD_EOF_RETRY_MAX"] = "128"
    # os.environ[
    #     "LD_PRELOAD"
    # ] = "/usr/lib/x86_64-linux-gnu/libcuda.so:/usr/lib/x86_64-linux-gnu/libnvcuvid.so:$LD_PRELOAD"

    vr = VideoReader(str(path), num_threads=1, ctx=gpu(0))
    vr[10].asnumpy()
    vr.seek(0)
    dtype = np.dtype(dtype)

    shm = SharedMemory(name=shm_name)
    buf = np.ndarray(frame_shape, dtype=dtype, buffer=shm.buf)

    global interrupt

    while True:
        request = request_queue.get(block=True)
        if request is None:
            break

        # if the interrupt signal is received it will decode the next frame and skip this one
        if interrupt:
            interrupt = False
            continue

        rid, index = request
        frame = vr[index].asnumpy()

        # if the interrupt signal is received it will skip the copy step to save some time
        # this allows it to start decoding the next frame sooner
        if interrupt:
            interrupt = False
            continue

        if frame.shape != buf.shape or frame.dtype != dtype:
            shm.close()
            shm = SharedMemory(create=True, size=frame.nbytes)
            buf = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
            dtype = frame.dtype

        # if the interrupt signal is received it will decode the next frame and skip this one
        if interrupt:
            interrupt = False
            continue

        np.copyto(buf, frame)
        response_queue.put((rid, frame.shape, str(frame.dtype), shm.name))

    shm.close()
