#!usr/bin/env python
from __future__ import print_function
from builtins import range
from builtins import object
__author__ = "Hosik Chae"
__email__ = "CKMagenta@gmail.com"
__copyright__ = "Copyright 2017 RoMeLa"
__date__ = "10/30/2017"

import time

def wait(duration, t0=None, discount=0.1):
    if t0 is None:
        t0 = time.time()

    t1 = time.time()
    while (t1 - t0 < duration):
        time.sleep(discount * (duration - t1 + t0))
        t1 = time.time()


class RepeatedTimer(object):
    def __init__(self, interval, function, auto_start=True, daemon=True, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.daemon     = daemon

        self.args       = args
        self.kwargs     = kwargs
        for k, v in list(kwargs.items()):
            setattr(self, k, v)

        self.is_running = False
        if auto_start is True:
            self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.daemon = self.daemon
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


class Time(object):
    def __init__(self, pool="pool.ntp.org"):
        self.time_offset = 0.0
        self.ntp_client = ntplib.NTPClient()
        self.ntp_pool = pool
        self.ntp_last_response = None

    def sync_time(self, pool=None):
        pool = pool or self.ntp_pool
        try:
            self.ntp_last_response = self.ntp_client.request(pool)
        except Exception as e:
            print("An exception of type {0} occurred. Arguments:\n{1!r}".format(type(e).__name__, e.args))
            raise Time.TimeException

        self.time_offset = self.ntp_last_response.offset

        return self

    def get_time(self, time=None):
        if time is None:
            time = time.time()

        return time + self.time_offset

    def get_network_time(self):
        if self.ntp_last_response is None:
            self.sync_time()
        return self.ntp_last_response.tx_time

    def __str__(self):
        return time.ctime(self.get_time())

    class TimeException(Exception):
        pass

class SyncTimer(object):
    t0 = 0
    t1 = 0

    def __init__(self, dt):
        self.dt = dt

    def start(self, t0 = None):
        if t0 == None:
            self.t0 = time.time()
            self.t1 = self.t0
        else:
            self.t0 = t0
            self.t1 = t0

        return self.t0

    def set_dt(self, dt):
        self.dt = dt

    def set_next_timestep(self, t1):
        self.t1 = t1 + self.t0

    def to_next_timestep(self, n_step = 1):
        self.t1 += self.dt * n_step

    def get_time_elapsed(self):
        return time.time() - self.t0

    def wait_until_next_timestep(self, to_next_n_timestep=None):
        if time.time() < self.t1:
            return True
        else:
            if to_next_n_timestep != None:
                self.to_next_timestep(to_next_n_timestep)
            return False

    @classmethod
    def sleep(cls, t, dt, simulator_step_advance=None):
        t0 = time.time()
        if simulator_step_advance == None:
            while (time.time() - t0 < t + dt) :
                pass
        else:
            for idx in range(int(t//dt)):
                simulator_step_advance()

