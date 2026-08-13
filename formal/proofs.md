# TxnOpt T1-T4 proof obligations

Status: T1/T2 inductive proof plus bounded TLC model; T3/T4 review draft.

## T1: completion-order independence under fixed work

Assume a fixed initial state, random tape, canonical candidate sequence
`C = <c1, ..., cn>`, deterministic Oracle results, and a commit rule that may
commit only the least unresolved candidate index. Physical workers may complete
any pending index.

Inductive invariant `I(k)`: after `k` commits, the visible semantic trace is
exactly `<c1, ..., ck>`, the visible cache is exactly the set of results for that
prefix, and no result with index greater than `k` is visible.

- Base case `k = 0`: the trace and cache are empty.
- Physical completion step: it changes only the private completed set, so
  `I(k)` is preserved for every completion permutation.
- Commit step: the guard permits only index `k + 1`; appending `c(k + 1)` and
  atomically publishing its cache entries establishes `I(k + 1)`.
- Termination step: it changes neither trace nor cache.

Therefore every physical completion order refines to the same canonical commit
trace. `TxnOpt.tla` checks the corresponding invariant for four candidates,
including every completion/failure interleaving.

## T2: deadline and worker failure return a commit prefix

The only visible-state action is `Commit`. Deadline, worker failure, validation
failure, stale snapshot, and cache publication failure take `Terminate`,
`ABORTED`, or `INTERRUPTED` without changing committed state or cache. By T1's
invariant, the state before failure is a canonical prefix; a failure action
preserves it, and terminated states have no enabled commit action. Thus the
returned state is the last complete commit prefix. The Python fault tests and
native typed receipt are the refinement witnesses.

## T3: unconditional trilemma counterexample

Consider a state-dependent search with initial state `s0` and two logical
candidates. Candidate `c2` is generated from the state produced by resolving
`c1`. Choose an instance where `c1` is accepted and changes `s0` to `s1`, and
where evaluating the version of `c2` generated from `s0` is not equivalent to
evaluating it from `s1`.

Before `c1` resolves, an implementation has three exhaustive choices:

1. Do not start `c2`: serial semantics and zero waste hold, but fully
   asynchronous progress does not.
2. Start `c2(s0)`: asynchronous progress holds; if `c1` is accepted, that work
   must be discarded, so zero speculative waste fails.
3. Make `c2(s0)` visible despite accepting `c1`: asynchronous progress and zero
   discard may hold, but the result is not the serial `c2(s1)` result, so
   schedule independence fails.

Hence schedule independence, zero speculative waste, and fully asynchronous
progress cannot all hold unconditionally for arbitrary state-dependent search.

## T4: bounded speculative waste

Let `W` be the maximum number of unresolved speculative candidates, `Qmax` the
maximum requests per candidate, `Cmax` an upper bound on one request's cost,
`Bremaining` the remaining fixed-work budget at the boundary, and `P` the
physical parallelism.

At most `W * Qmax` speculative requests can exist. Budget admission additionally
limits started speculative requests to `Bremaining`, so discarded work units are
at most `min(Bremaining, W * Qmax)`. Multiplication by non-negative `Cmax` gives
the discarded-cost bound. After the boundary, at most `P` requests can still be
executing, while the same window bound applies; therefore post-boundary in-flight
cost is at most `min(P, W * Qmax) * Cmax`.

The executable function `txnopt._internal.waste_bounds.bounded_waste` evaluates
these expressions. A performance claim additionally requires measured finite
`Qmax` and `Cmax`; otherwise the algebraic statement is rejected as vacuous.
