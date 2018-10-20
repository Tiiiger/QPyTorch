class Number():
    def __init__(self):
        pass

    def __str__(self):
        raise NotImplemented

class FixedPoint(Number):
    def __init__(self, wl, fl):
        assert wl > 0, "wl > 0"
        assert fl > 0, "fl > 0"
        self.wl = wl
        self.fl = fl

    def __str__(self):
        return "FixedPoint Number with wl={:d}, fl={:d}".format(wl, fl)

class BlockFloatingPoint(Number):
    def __init__(self, wl):
        assert wl > 0, "wl > 0"
        self.wl = wl

    def __str__(self):
        return "BlockFloatingPoint Number with wl={:d}".format(wl)

class FloatingPoint(Number):
    def __init__(self, exp, man):
        assert exp > 0, "exp > 0"
        assert man > 0, "man > 0"
        assert exp <= 8, "exp <= 8"
        assert man <= 23, "man <= 23"
        self.exp = exp
        self.man = man

    def __str__(self):
        return "FloatingPoint Number with exponent={:d}, mantissa={:d}".format(self.exp, self.man)