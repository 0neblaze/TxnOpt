# EVRPTW native case boundary

EVRPTW context packing, safe screening, and exact-charging are exposed behind
`txnopt-native-round-v1`. During Level 1 they reuse the frozen, differentially
tested numeric headers from their historical paths; the active wheel does not
install or expose the old ABI.

This structural directory does not rename or wrap the frozen Stage 5.2 ABI.
