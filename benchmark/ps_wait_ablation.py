"""Benchmark-only reproduction of the former PS polling loop.

The environment switch is inherited by Windows spawn children. Production
entry points never import this module. Keep the normal model, transport,
optimizer, validation and recorder unchanged.
"""

import os
import time

from MyFlows.distributed.constants import STATUS_WAITING
from MyFlows.distributed.engine import PSEngine


def polling_wait(self, once, message, wait_s):
    deadline = time.monotonic() + max(0.0, float(wait_s))
    while True:
        reply = once(message)
        if reply.get('status') != STATUS_WAITING or time.monotonic() >= deadline:
            return reply
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))


if os.environ.get('MYFLOWS_BENCHMARK_PS_WAIT') == 'poll20ms':
    PSEngine._poll = polling_wait


if __name__ == '__main__':
    from benchmark.distributed_experiment import main
    main()
