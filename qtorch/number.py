__all__ = ['Number', 'FixedPoint', 'BlockFloatingPoint', 'FloatingPoint']

class Number():
    def __init__(self):
        pass

    def __str__(self):
        raise NotImplemented

class FixedPoint(Number):
    def __init__(self, wl, fl):
        assert wl > 0, "invalid bits for word length: {}".format(wl)
        assert fl > 0, "invalid bits for fractional length: {}".format(fl)
        self.wl = wl
        self.fl = fl

    def __str__(self):
        return "FixedPoint Number with wl={:d}, fl={:d}".format(self.wl, self.fl)

class BlockFloatingPoint(Number):
    def __init__(self, wl):
        assert wl > 0, "invalid bits for word length:{}".format(self.wl)
        self.wl = wl

    def __str__(self):
        return "BlockFloatingPoint Number with wl={:d}".format(self.wl)

class FloatingPoint(Number):
    def __init__(self, exp, man):
        assert 8 >= exp > 0, "invalid bits for exponent:{}".format(exp)
        assert 23 >= man > 0, "invalid bits for mantissa:{}".format(man)
        self.exp = exp
        self.man = man

    def __str__(self):
        return "FloatingPoint Number with exponent={:d}, mantissa={:d}".format(self.exp, self.man)
