# TxnOpt native core boundary

This directory contains the active `txnopt-native-round-v1` pybind façade and
generic round/concurrency ownership. Budget reservation, cache publication,
commit/rollback, and semantic trace ownership remain in `TxnRuntime`; the
native module returns one prepared typed receipt and never publishes Python
state or cache.

The frozen `stage05.2-*` sources and exports remain in their historical paths.
Level 1 does not implement a multi-round full-native fast path.
