# TxnOpt native core boundary

This directory contains the active `txnopt-native-round-v1` pybind façade and
the deep prepared-round module. One call now owns the native phase machine,
complete-work reservation, ordered evaluation settlement, prepared cache delta,
semantic phase trace, and concurrency. Python `TxnRuntime` still owns the
solve-wide budget and the only publishable state/cache commit; the native
module returns one prepared typed receipt and never publishes Python state or
cache.

The frozen `stage05.2-*` sources and exports remain in their historical paths.
Level 1 does not implement a multi-round full-native fast path.
