class DelayError(Exception):
    def __init__(self, delay, message="Delay should be strictly greater than 0"):
        self.delay = delay
        self.message = message
        super().__init__(self.message)


class FrameStackingError(Exception):
    def __init__(
        self,
        delay,
        message="Number of stacked frames should be strictly greater than 0",
    ):
        self.delay = delay
        self.message = message
        super().__init__(self.message)
