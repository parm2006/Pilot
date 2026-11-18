"""Primary-eye blink gestures shared by both cursor scripts."""


class WinkDetector:
    def __init__(self, close, click_frames=10, pause_hold=1.2):
        self.close = close
        self.click_frames = click_frames
        self.pause_hold = pause_hold
        self.clear()

    def clear(self):
        self.since = None
        self.frames = 0
        self.closed = False
        self.pause_fired = False

    def update(self, t, primary, paused=False):
        # A lower release threshold keeps small score fluctuations from ending a hold.
        threshold = self.close * (0.7 if self.closed else 1.0)
        if primary < threshold:
            self.clear()
            return []

        self.closed = True
        if self.since is None:
            self.since = t
        self.frames += 1
        if self.pause_fired:
            return []
        if t - self.since >= self.pause_hold:
            self.pause_fired = True
            return ["toggle_pause"]
        if not paused and self.frames % self.click_frames == 0:
            return ["left"]
        return []

    def frozen(self, t):
        return self.closed and self.frames >= self.click_frames
