class DelayError(Exception):
    def __init__(self, delay, message="Delay should be strictly greater than 0"):
        self.delay = delay
        self.message = message
        super().__init__(self.message)
