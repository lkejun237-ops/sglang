# SPDX-License-Identifier: Apache-2.0
"""Linux benchmark launcher ownership, including children of an exited main."""

import contextlib
import ctypes
import errno
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def owned_popen(command, **kwargs):
    """Start a subreaper before the command can create any descendants."""
    if (
        sys.platform != "linux"
        or kwargs.get("start_new_session", True) is not True
        or kwargs.get("shell")
        or kwargs.get("pass_fds")
    ):
        raise RuntimeError("Owned benchmark processes require a new Linux session")
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    try:
        os.set_blocking(read_fd, False)
        kwargs["pass_fds"] = (write_fd,)
        kwargs["start_new_session"] = True
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(write_fd),
                *map(str, command),
            ],
            **kwargs,
        )
    except BaseException:
        os.close(read_fd)
        raise
    finally:
        os.close(write_fd)
    process._zing_cleanup_fd = read_fd
    process._zing_cleanup_receipt = None
    return process


def verify_owned_cleanup(process):
    """Only a receipt from the original supervisor can confirm child cleanup."""
    if not hasattr(process, "_zing_cleanup_fd"):
        return
    if process.poll() is None:
        raise RuntimeError("Owned supervisor is still running")
    if process._zing_cleanup_fd is not None:
        descriptor, process._zing_cleanup_fd = process._zing_cleanup_fd, None
        try:
            data = bytearray()
            while len(data) <= 4096:
                block = os.read(descriptor, 4097 - len(data))
                if not block:
                    break
                data.extend(block)
            if len(data) > 4096:
                raise ValueError("Owned cleanup receipt exceeds its size limit")
            process._zing_cleanup_receipt = json.loads(data)
        except (OSError, ValueError):
            process._zing_cleanup_receipt = None
        finally:
            os.close(descriptor)
    if not (process._zing_cleanup_receipt or {}).get("complete"):
        raise RuntimeError("Owned supervisor did not confirm descendant cleanup")


def _reap(main):
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return True
        except OSError as error:
            if error.errno == errno.EINTR:
                continue
            raise
        if not pid:
            return False
        if main is not None and pid == main.pid and main.returncode is None:
            main.returncode = os.waitstatus_to_exitcode(status)


def _cleanup(main, timeout=30):
    import psutil

    owner = psutil.Process()
    began = time.monotonic()
    signalled = set()
    while not _reap(main):
        now = time.monotonic()
        if now - began >= timeout:
            raise TimeoutError("Owned descendants remain after cleanup deadline")
        for child in owner.children(recursive=True):
            # Process objects retain creation identity; never resolve an old PID
            # after reaping or enumerate unrelated processes by process group.
            with contextlib.suppress(psutil.NoSuchProcess):
                if child.status() == psutil.STATUS_ZOMBIE:
                    continue
                if now - began >= 20:
                    child.kill()
                elif child not in signalled:
                    child.terminate()
                    signalled.add(child)
        time.sleep(0.02)


def _main(descriptor, command):
    main = None
    requested_signal = 0
    original_code = None
    cleanup_complete = False
    began = time.monotonic()

    def request_stop(number, frame):
        nonlocal requested_signal
        requested_signal = requested_signal or number

    try:
        if sys.platform != "linux":
            raise RuntimeError("Owned benchmark supervision requires Linux")
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0):  # PR_SET_CHILD_SUBREAPER, before spawn.
            raise OSError(ctypes.get_errno(), "Cannot enable child subreaper")
        for number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(number, request_stop)
        main = subprocess.Popen(command)
        while not requested_signal:
            original_code = main.poll()
            if original_code is not None:
                break
            time.sleep(0.02)
    except BaseException as error:
        print(f"OWNED_PROCESS_ERROR {type(error).__name__}: {error}", file=sys.stderr)
        original_code = original_code or 125
    finally:
        cleanup_began = time.monotonic()
        try:
            _cleanup(main)
            cleanup_complete = True
        except BaseException as error:
            print(
                f"OWNED_CLEANUP_ERROR {type(error).__name__}: {error}", file=sys.stderr
            )
        if original_code is None and main is not None:
            original_code = main.returncode
        receipt = dict(
            complete=cleanup_complete,
            lifetime_seconds=time.monotonic() - began,
            cleanup_seconds=time.monotonic() - cleanup_began,
            main_returncode=original_code,
            requested_signal=requested_signal,
        )
        try:
            os.write(descriptor, json.dumps(receipt).encode())
        finally:
            os.close(descriptor)
        print("OWNED_PROCESS_CLEANUP", json.dumps(receipt), file=sys.stderr, flush=True)
    code = original_code or (128 + requested_signal if requested_signal else 0)
    if code < 0:
        code = 128 - code
    return code if cleanup_complete or code else 125


if __name__ == "__main__":
    raise SystemExit(_main(int(sys.argv[1]), sys.argv[2:]))
