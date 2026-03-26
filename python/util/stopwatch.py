import time
from typing import Self


class Stopwatch:
    """Measures elapsed wall-clock time between explicit start and stop calls."""

    def __init__(self):
        """Initializes the stopwatch without any recorded timestamps."""

        self.start_time = None
        self.stop_time = None

    def start(self) -> Self:
        """Starts the stopwatch and returns the instance."""

        self.start_time = time.perf_counter()
        return self

    def stop(self):
        """Stops the stopwatch and returns the instance."""

        self.stop_time = time.perf_counter()
        return self

    def duration_seconds_float(self) -> float:
        """Calculates the elapsed duration in seconds as a float.

        Returns:
            The elapsed duration between the start and stop times in seconds.

        Raises:
            ValueError: The stopwatch was not started and stopped properly.
        """

        if self.start_time is None or self.stop_time is None:
            raise ValueError(
                'Stopwatch has not been started and stopped properly.'
            )

        return self.stop_time - self.start_time

    def duration_seconds(self) -> int:
        """Calculates the elapsed duration in whole seconds.

        Returns:
            The elapsed duration between the start and stop times in seconds.
        """

        return int(self.duration_seconds_float())

    def duration_milliseconds(self) -> int:
        """Calculates the elapsed duration in whole milliseconds.

        Returns:
            The elapsed duration between the start and stop times in milliseconds.
        """

        return int(self.duration_seconds_float() * 1000)
