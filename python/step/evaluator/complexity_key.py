def ridge_regression_complexity_key(
    hp: list[float],
) -> tuple[float, ...]:
    """Returns the complexity key for ridge regression.

    Args:
        hp: A list of hyperparameter values ordered as ``[degree, alpha]``.

    Returns:
        A tuple where lower values correspond to simpler models. Lower
        polynomial degree is simpler, and higher ridge regularization
        (``alpha``) is simpler.
    """

    return hp[0], -hp[1]


def gaussian_process_regression_complexity_key(
    hp: list[float],
) -> tuple[float, ...]:
    """Returns the complexity key for Gaussian process regression.

    Args:
        hp: A list of hyperparameter values ordered as
            ``[length_scale, noise_level]``.

    Returns:
        A tuple where lower values correspond to simpler models. Larger
        ``length_scale`` produces smoother functions, and larger
        ``noise_level`` attributes more variation to noise; both are treated
        as simpler.
    """

    return (-hp[0], -hp[1])


def support_vector_regression_complexity_key(
    hp: list[float],
) -> tuple[float, ...]:
    """Returns the complexity key for support vector regression.

    Args:
        hp: A list of hyperparameter values ordered as ``[C, epsilon]`` or
            ``[C, epsilon, gamma]``.

    Returns:
        A tuple where lower values correspond to simpler models. Lower ``C``
        is simpler, higher ``epsilon`` is simpler, and when tuned, lower
        ``gamma`` is simpler.
    """

    return (hp[0], -hp[1], hp[2]) if len(hp) == 3 else (hp[0], -hp[1])
