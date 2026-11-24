from typing import Dict

import numpy as np


class Grid:
    def __init__(self, h: float, start: float, end: float, fill_type="arange") -> None:
        """
        Args:
            h (float): Step size
            x1 (float): Start of interval
            x2 (float): End of interval
        """
        self.fill_type = fill_type
        self._start = start
        self._end = end
        self.h = h

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def points(self):
        return self._points

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, val):
        if self.fill_type == 'arange':
            self._h = val
        else:
            self._n = int(np.floor((self._end - self._start) / val) + 1)
        self._grid_setup()

    def n(self, val):
        self._n = val
        self._grid_setup()

    def _grid_setup(self):
        if self.fill_type == 'arange':
            self._points = np.arange(self._start, self._end + 1e-8, self._h)
        else:
            self._points = np.linspace(self._start, self._end, self._n)
            self._h = self._points[1] - self._points[0]

    def __len__(self) -> int:
        return len(self._points)

    def metadata(self) -> Dict:
        return {
            "h": self.h,
            "start": self.start,
            "end": self.end,
        }
