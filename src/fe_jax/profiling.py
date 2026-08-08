import threading
import psutil
import time
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import re
import inspect
import os
import json
import jax
from typing import Callable
import statistics
from contextlib import contextmanager
from functools import wraps


def pack(**kwargs):
    """
    Helper to transform arguments into a dictionary. Use with the timeit function.
    """
    return kwargs 

def timer(time_jit=False, n_calls=1):
    def timer_decorator(f):
        """
        Prints how long a function took after each call.

        Note: DO NOT use this with function that modify inputs since it calls the function multiple
        times to get a good average.

        Inspired by: https://stackoverflow.com/a/27737385
        """

        @wraps(f)
        def wrap(*args, **kw):
            if time_jit:
                start = time.perf_counter()
                result = f(*args, **kw)
                first_call_time = time.perf_counter() - start

                start = time.perf_counter()
                for i in range(n_calls):
                    result = f(*args, **kw)
                loop_avg_time = (time.perf_counter() - start) / n_calls
                jit_time = first_call_time - loop_avg_time

                print(
                    f'Call to "{f.__name__}" JIT took {jit_time:.6f} sec and calls took {loop_avg_time:.6f} sec / call (avg {n_calls})'
                )

            else:
                start = time.perf_counter()
                for i in range(n_calls):
                    result = f(*args, **kw)
                loop_avg_time = (time.perf_counter() - start) / n_calls

                print(
                    f'Call to "{f.__name__}" took {loop_avg_time:.6f} sec / call (avg {n_calls})'
                )

            return result

        return wrap

    return timer_decorator


def timeit(
    f: Callable,
    fixed_kwargs: dict,
    generated_kwargs: dict,
    time_jit: bool = True,
    n_calls: int = 1,
    timings_figure_filepath: str = "",
    return_timing = False,
    return_memory = False
):
    """
    Times a function call, possibly timing just-in-time compilation on the first call and takes
    the average of multiple calls.

    Parameters
    ----------
    f             : function
        Function to be measured
    fixed_kwargs  : dict[str,Any]
        Dictionary of keyword arguments with fixed values
    generated_kwargs : dict[str,function]
        Dictionary of keyword arguments with values that should be generated each call
    time_jit      : bool
        Indicates if the first call should be measured separately (for JIT)
    n_calls       : int
        Number of times to call the function and average

    Returns
    -------
    Result of f(args)
    """
    if time_jit:
        kwargs = fixed_kwargs | {arg: func() for arg, func in generated_kwargs.items()}
        start = time.perf_counter()
        result = jax.block_until_ready(f(**kwargs))
        first_call_time = time.perf_counter() - start

        if return_memory:
            init_memory_profile()
        times = [0.0] * n_calls
        for i in range(n_calls):
            kwargs = fixed_kwargs | {
                arg: func() for arg, func in generated_kwargs.items()
            }
            start = time.perf_counter()
            result = jax.block_until_ready(f(**kwargs))
            times[i] = time.perf_counter() - start
        loop_avg_time = sum(times) / n_calls
        jit_time = first_call_time - loop_avg_time
        if return_memory:
            memory_usage = get_memory_profile()

        print(
            f'Call to "{f.__name__}" JIT took {jit_time:.6f} sec and calls took {loop_avg_time:.6f} sec / call (n_calls: {n_calls} min: {min(times):.6f} max: {max(times):.6f} std-dev: {statistics.stdev(times):.6f})'
        )

    else:
        if return_memory:
            init_memory_profile()
        times = [0.0] * n_calls
        for i in range(n_calls):
            kwargs = fixed_kwargs | {
                arg: func() for arg, func in generated_kwargs.items()
            }
            start = time.perf_counter()
            result = jax.block_until_ready(f(**kwargs))
            times[i] = time.perf_counter() - start
        loop_avg_time = sum(times) / n_calls
        jit_time = 0.0
        first_call_time = 0.0
        if return_memory:
            memory_usage = get_memory_profile()

        print(
            f'Call to "{f.__name__}" took {loop_avg_time:.6f} sec / call (n_calls: {n_calls} min: {min(times):.6f} max: {max(times):.6f} std-dev: {statistics.stdev(times):.6f})'
        )

    if timings_figure_filepath != "":
        import matplotlib.pyplot as plt

        # times.insert(0, first_call_time)
        plt.plot(times, label="calls")
        plt.hlines(
            [loop_avg_time],
            xmin=0,
            xmax=len(times) - 1,
            linestyles="--",
            label="average",
        )
        plt.xlabel("Call Index")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.savefig(timings_figure_filepath)
        plt.clf()

    # dynamic return values
    to_return = [result]
    if return_timing:
        to_return.extend([times, jit_time, first_call_time])
    if return_memory:
        to_return.extend([memory_usage.get('peak_memory', {})])
    return tuple(to_return)


def get_colors_from_cmap(cmap_name, num_colors):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(x) for x in np.linspace(0, 1, num_colors)]


def get_current_pid_host_memory():
    """
    Returns current memory used on host in MB.
    """
    return psutil.Process().memory_info().rss / 1_000_000


class CPUPoll:
    def __init__(self, sampling_time=0.1):
        self.process = psutil.Process()
        self.poll_thread = threading.Thread(target=self.__poll_cpu)
        self.stop_flag = False
        self.kill_flag = False
        self.sampling_time = sampling_time
        self.restart()
        self.poll_thread.start()

    def __del__(self):
        self.kill_flag = True
        self.poll_thread.join()

    def mark_event(self, event_name: str):
        if not self.stop_flag:
            self.event_times.append(time.time() - self.start_time)
            self.event_names.append(event_name)

    def restart(self):
        self.stop_flag = True
        time.sleep(1.2 * self.sampling_time)

        # Continuous poll data
        self.times = []
        self.cpu_percent = []
        self.memory = []

        # Event data
        self.event_times = []
        self.event_names = []

        self.start_time = time.time()
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True
        time.sleep(1.2 * self.sampling_time)

    def get_plt_fig(self, ax=None, legend=True):

        if ax is None:
            fig, ax1 = plt.subplots()
        else:
            ax1 = ax

        ax1.plot(self.times, self.cpu_percent, "g-", label="cpu usage")
        ax1.set_xlabel("Elapsed Time (s)")
        ax1.set_ylabel(f"Total Usage of {psutil.cpu_count()} CPUs (%)", color="g")
        ax1.set_ylim(0.0, 100.0)

        colors = get_colors_from_cmap(
            "viridis", len(self.event_times)
        )  # Get 10 colors from the 'viridis' colormap
        for i, (t, n) in enumerate(zip(self.event_times, self.event_names)):
            ax1.vlines(
                x=t, ymin=0.0, ymax=100.0, colors=colors[i], linestyles="--", label=n
            )

        ax2 = ax1.twinx()
        ax2.plot(self.times, self.memory, "b-", label="memory")
        ax2.set_ylabel("Memory Usage (MB)", color="b")
        ax2.set_ylim(0.0, 1.1 * max(self.memory))

        if legend:
            ax1.legend()

        return (ax1, ax2)

    def __poll_cpu(self):
        while not self.kill_flag:
            if not self.stop_flag:
                self.times.append(time.time() - self.start_time)
                self.cpu_percent.append(
                    self.process.cpu_percent(interval=0.1) / psutil.cpu_count()
                )
                self.memory.append(self.process.memory_info().rss / 1_000_000)
            time.sleep(self.sampling_time)


@contextmanager
def poll_cpu():
    try:
        cpu_poll = CPUPoll()
        yield cpu_poll
    finally:
        cpu_poll.__del__()


class MemoryProfile:
    def __init__(self, prof_file, base=None):
        self.base_prof_file = base
        self.prof_file = prof_file
        self.allocations = []
        self.locations = {}
        self.mappings = {}

        # optionally subtract a base
        if self.base_prof_file is not None:
            cmd = f"pprof -raw --base {self.base_prof_file} {self.prof_file}"
        else:
            cmd = (f"pprof -raw {self.prof_file}",)

        # get raw input
        raw_prof = subprocess.run(
            cmd, shell=True, capture_output=True, text=True
        ).stdout

        # parse
        iterable = iter(
            re.split(r"(allocations/count space/bytes|Locations|Mappings)", raw_prof)
        )
        while True:
            try:
                item = next(iterable)

                # parse allocations
                if item == "allocations/count space/bytes":
                    tmp = next(iterable)
                    chunks = [
                        x for x in re.split(r"\n(?= +[-]{0,1}\d)", tmp) if x.strip()
                    ]
                    for chunk in chunks:
                        lines = chunk.split("\n")

                        # first line is count bytes : stack_trace
                        tmp = [
                            x.strip()
                            for x in re.sub(r"\s+", " ", lines[0].strip()).split(":")
                            if x
                        ]
                        count, bytes = [int(x) for x in tmp[0].split() if x]
                        stack_trace = [int(x) for x in tmp[1].split() if x]

                        # second line is info
                        kind = None
                        device = None
                        for tmp in lines[1].strip().split():
                            k, v = tmp.split(":", 1)
                            v = re.sub(r"(\[|\])", "", v)
                            if k == "kind":
                                kind = v
                            elif k == "device":
                                device = v

                        # create entry
                        d = dict(
                            count=count,
                            bytes=bytes,
                            stack_trace=stack_trace,
                            kind=kind,
                            device=device,
                        )
                        self.allocations.append(d)

                # stack trace entries
                elif item == "Locations":
                    tmp = next(iterable)
                    lines = [x for x in tmp.split("\n") if x]
                    for line in lines:
                        k, v = line.strip().split(":", 1)
                        self.locations[int(k)] = v.strip()

                # mappings
                elif item == "Mappings":
                    tmp = next(iterable)
                    lines = [x for x in tmp.split("\n") if x]
                    for line in lines:
                        k, v = line.strip().split(":", 1)
                        self.mappings[int(k)] = v.strip()

            except StopIteration:
                break

    def get_device_memory(self, device):
        device = None if device == "cpu" else device
        total = 0  # bytes
        for allocation in self.allocations:
            if allocation["device"] == device:
                total += allocation["count"] * allocation["bytes"]
        return total


def init_memory_profile():

    # clear out the log file
    prof_log = os.path.join("prof", f"memory_profile.log")
    with open(prof_log, "w") as f:
        pass


def get_memory_profile():

    # read data from the log file
    prof_log = os.path.join("prof", f"memory_profile.log")
    with open(prof_log, "r") as f:
        results = [json.loads(line.strip()) for line in f.readlines()]

    # find peak memory usage for each device
    peak_memory = {str(device): dict(bytes=0, label="") for device in jax.devices()}
    for entry in results:
        label, data = next(iter(entry.items()))
        for device, bytes in data.items():
            d = peak_memory[device]
            if bytes > d["bytes"]:
                d["bytes"] = bytes
                d["label"] = label

    ## print results
    # print('Peak memory usage per device (bytes):')
    # for device, d in peak_memory.items():
    #    print(f'  {device:8s} {str(d["bytes"]):>8s} ({d["label"]})')

    return {"peak_memory": peak_memory}


def start_memory_profile(label=None):
    if label is None:
        # get the name of the calling function
        label = inspect.stack()[1].function

    # define the profile output file
    prof_file_0 = os.path.join("prof", f"memory_profile_{label}_0.prof")

    # write memory profile
    jax.profiler.save_device_memory_profile(prof_file_0)


def stop_memory_profile(label=None):
    if label is None:
        # get the name of the calling function
        label = inspect.stack()[1].function

    # define the profile output files
    prof_file_0 = os.path.join("prof", f"memory_profile_{label}_0.prof")
    prof_file_1 = os.path.join("prof", f"memory_profile_{label}_1.prof")

    # write memory profile
    jax.profiler.save_device_memory_profile(prof_file_1)

    # parse memory profile and subtract original
    pprof = MemoryProfile(prof_file_1, base=prof_file_0)

    # get device memory
    results = {
        label: {
            str(device): pprof.get_device_memory(str(device))
            for device in jax.devices()
        }
    }

    # write results
    prof_log = os.path.join("prof", f"memory_profile.log")
    with open(prof_log, "a") as f:
        f.write(f"{json.dumps(results)}\n")
