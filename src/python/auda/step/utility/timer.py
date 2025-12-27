import time
from typing import Dict, override

from auda.step.spec import Spec
from auda.utils.pipeline import IOValueMap, Pipeline, Step, step


@step(
    id='UT-TIMER',
    description='',
    input_specs=[
        Spec.TIMER_NUMBER.optional(0),
        Spec.START_TIME_MAP.optional(),
        Spec.TIMER_PRINT_CALLBACK_SET.optional(False),
    ],
    output_specs=[Spec.START_TIME_MAP, Spec.TIMER_PRINT_CALLBACK_SET],
)
class Timer(Step):
    @override
    def run(
        self,
        timer_number: int,
        start_time_map: Dict[int, float] | None,
        timer_print_callback_set: bool,
    ) -> IOValueMap:
        if start_time_map is None:
            start_time_map = {}

        start_time_map[timer_number] = time.perf_counter()

        if not timer_print_callback_set:

            def print_timers(_: Pipeline) -> None:
                current_time = time.perf_counter()

                print('--- Timer Results ---')
                for timer_number, start_time in start_time_map.items():
                    elapsed_ms = (current_time - start_time) * 1000
                    print(f'[Timer {timer_number}] {elapsed_ms:.3f} ms')

            self.pipeline.schedule(print_timers)

        return {
            Spec.START_TIME_MAP.name: start_time_map,
            Spec.TIMER_PRINT_CALLBACK_SET.name: True,
        }
