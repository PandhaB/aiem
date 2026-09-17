"""Choose a native-scale network size when inference is allowed larger than training."""

from __future__ import annotations

import math


def longest_side(image_sizes: list[tuple[int, int]] | None) -> int | None:
    if not image_sizes:
        return None
    return max(max(width, height) for width, height in image_sizes)


def native_capped_size(
    image_sizes: list[tuple[int, int]] | None,
    max_dimension: int,
    *,
    step: int,
    min_value: int,
) -> int:
    """Longest image side, rounded to ``step``, never above ``max_dimension``.

    Empty/missing images use ``max_dimension`` itself. Rounding up is skipped when
    it would exceed the cap, so a user limit of 1000 does not become 1024.
    """
    cap = int(max_dimension)
    if cap < min_value:
        raise ValueError(f"Maximum image dimension must be at least {min_value}.")
    if step < 1:
        raise ValueError("step must be at least 1.")
    longest = longest_side(image_sizes)
    raw = min(longest, cap) if longest else cap
    rounded = max(step, math.ceil(raw / step) * step)
    if rounded > cap:
        rounded = (cap // step) * step
    return max(min_value, rounded)
