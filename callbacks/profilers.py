from pytorch_lightning.profilers.simple import SimpleProfiler, _TABLE_DATA_EXTENDED
import time
import torch
from typing import Dict, List, Optional, Tuple, Union


class SimpleProfilerCustom(SimpleProfiler):
    def __init__(self, *args, burn_in_dict={}, **kwargs):
        """
        burn_in_dict:   for those specific actions, we will not record the first `burn_in_dict[action_name]` calls
        """
        super().__init__(*args, **kwargs)
        self.burn_in_dict = burn_in_dict

    def _make_report_extended(self) -> Tuple[_TABLE_DATA_EXTENDED, float, float]:
        total_duration = time.monotonic() - self.start_time
        report = []

        for a, d in self.recorded_durations.items():
            if a in self.burn_in_dict:
                d = d[self.burn_in_dict[a]:]
            d_tensor = torch.tensor(d)
            len_d = len(d)
            sum_d = torch.sum(d_tensor).item()
            percentage_d = 100.0 * sum_d / total_duration

            report.append((a, sum_d / len_d, len_d, sum_d, percentage_d))

        report.sort(key=lambda x: x[4], reverse=True)
        total_calls = sum(x[2] for x in report)
        return report, total_calls, total_duration

    def _make_report(self):
        raise NotImplementedError("This method is not implemented in the custom profiler")