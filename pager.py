
class Pager(object):
    def __init__(self, count):
        self.count = count
        self.current = 0

    @property
    def next(self):
        n = self.current + 1
        if n > self.count-1:
            n = n- self.count
        return n

    @property
    def prev(self):
        n = self.current - 1
        if n < 0 :
            n = n + self.count
        return n
