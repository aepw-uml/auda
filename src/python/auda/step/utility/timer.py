import time
from typing import Dict, override

from auda.step.spec import Spec
from auda.utils.pipeline import IOValueMap, Pipeline, Step, step


@step(
    id='UT-TIMER-START',
    description='',
    input_specs=[
        Spec.TIMER_NUMBER.optional(0),
        Spec.START_TIME_MAP.optional(),
        Spec.TIMER_PRINT_CALLBACK_SET.optional(False),
    ],
    output_specs=[Spec.START_TIME_MAP, Spec.TIMER_PRINT_CALLBACK_SET],
)
class TimerStart(Step):
    @override
    def run(
        self,
        timer_number: int,
        start_time_map: Dict[int, float] | None,
        timer_print_callback_set: bool,
    ) -> IOValueMap:
        if start_time_map is None:
            start_time_map = {}

        if timer_number in start_time_map:
            raise ValueError(
                f'Timer {timer_number} has already been started. '
                f'Please use a different timer number.'
            )

        start_time_map[timer_number] = time.perf_counter()

        if not timer_print_callback_set:

            def print_timers(pipeline: Pipeline) -> None:
                current_time = time.perf_counter()
                stop_time_map = pipeline.get_value(Spec.STOP_TIME_MAP.name)

                print('--- Timer Results ---')
                for timer_number, start_time in start_time_map.items():
                    if timer_number in stop_time_map:
                        stop_time = stop_time_map[timer_number]
                        elapsed_ms = (stop_time - start_time) * 1000
                    else:
                        elapsed_ms = (current_time - start_time) * 1000

                    print(f'[Timer {timer_number}] {elapsed_ms:.3f} ms')

            self.pipeline.schedule(print_timers)

        return {
            Spec.START_TIME_MAP.name: start_time_map,
            Spec.TIMER_PRINT_CALLBACK_SET.name: True,
        }


@step(
    id='UT-TIMER-STOP',
    description='',
    input_specs=[
        Spec.TIMER_NUMBER.optional(0),
        Spec.STOP_TIME_MAP.optional(),
    ],
    output_specs=[Spec.STOP_TIME_MAP],
)
class TimerStop(Step):
    @override
    def run(
        self,
        timer_number: int,
        stop_time_map: Dict[int, float] | None,
    ) -> IOValueMap:
        if stop_time_map is None:
            stop_time_map = {}

        stop_time_map[timer_number] = time.perf_counter()

        return {Spec.STOP_TIME_MAP.name: stop_time_map}
