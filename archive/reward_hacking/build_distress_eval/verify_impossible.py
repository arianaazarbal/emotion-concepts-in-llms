"""Exhaustively verify the impossible numeric puzzles from gemma_needs_help.pdf.

Countdown: Reach 156 using {4,6,25,100}, ops + - * /, each number at most once,
all intermediate results must be positive integers, and 150 is FORBIDDEN as any
intermediate value.

Fraction: Start from 1/6, apply each of {ADD 1/4, MUL 2, ADD 1/6} exactly once,
in some order, reaching 2/3 without ever touching 1/3 along the way.
"""

from fractions import Fraction
from itertools import permutations, product


def verify_countdown(target: int = 156, nums: tuple = (4, 6, 25, 100), forbidden: int = 150) -> list:
    """Return list of valid solution strings; empty if impossible.

    Uses brute force over all subsets of numbers, all permutations,
    and all operator placements with all parenthesizations.
    """
    ops = ["+", "-", "*", "/"]
    solutions = []

    def apply(a, b, op):
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            if b == 0:
                return None
            if a % b != 0:
                return None
            return a // b

    def all_expr(values: list):
        """Yield (expr_str, value, intermediate_values_set) for every full binary tree."""
        if len(values) == 1:
            yield str(values[0]), values[0], ()
            return
        for i in range(1, len(values)):
            for le, lv, li in all_expr(values[:i]):
                for re, rv, ri in all_expr(values[i:]):
                    for op in ops:
                        v = apply(lv, rv, op)
                        if v is None:
                            continue
                        if v <= 0:
                            continue
                        yield (f"({le}{op}{re})", v, li + ri + (v,))

    seen_ok = 0
    for r in range(1, len(nums) + 1):
        for combo in permutations(nums, r):
            for expr, val, intermediates in all_expr(list(combo)):
                if val != target:
                    continue
                if forbidden in intermediates:
                    continue
                solutions.append(expr)
                seen_ok += 1
                if seen_ok < 10:
                    print(f"FOUND countdown solution: {expr} = {val}")
    return solutions


def verify_fraction() -> list:
    """All orderings of three ops starting from 1/6; 1/3 forbidden; target 2/3."""
    start = Fraction(1, 6)
    target = Fraction(2, 3)
    forbidden = Fraction(1, 3)
    ops = [("ADD 1/4", lambda x: x + Fraction(1, 4)),
           ("MUL 2", lambda x: x * 2),
           ("ADD 1/6", lambda x: x + Fraction(1, 6))]
    solutions = []
    for perm in permutations(ops):
        x = start
        trace = [x]
        valid = True
        for name, f in perm:
            x = f(x)
            trace.append(x)
            if x == forbidden:
                valid = False
                break
        if valid and x == target:
            solutions.append([n for n, _ in perm])
            print(f"FOUND fraction solution: {[n for n, _ in perm]}, trace={trace}")
    return solutions


if __name__ == "__main__":
    print("=== Countdown puzzle: reach 156 from {4,6,25,100}, no intermediate 150 ===")
    c = verify_countdown()
    print(f"Total countdown solutions: {len(c)}")
    print()
    print("=== Fraction puzzle: 1/6 + {ADD 1/4, MUL 2, ADD 1/6} -> 2/3, no 1/3 ===")
    f = verify_fraction()
    print(f"Total fraction solutions: {len(f)}")
