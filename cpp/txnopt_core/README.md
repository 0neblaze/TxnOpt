# TxnOpt native core boundary

This directory is reserved for the new `txnopt-native-round-v1` implementation:
generic transaction state, budget accounting, cache transactions, concurrency,
and typed semantic/physical receipts.

The frozen `stage05.2-*` sources and exports remain in their historical paths.
No file is moved here until a Python/native round differential test exists.
Level 1 does not implement a multi-round full-native fast path.
