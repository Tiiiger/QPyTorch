__all__ = ['Number', 'FixedPoint', 'BlockFloatingPoint', 'FloatingPoint']

class Number():
    def __init__(self):
        pass

    def __str__(self):
        raise NotImplemented

class FixedPoint(Number):
    """
    Low-Precision Fixed Point Format.

    Args:
        - :attr: `wl` word length of each fixed point number
        - :attr: `fl` fractional length of each fixed point number
    """
    def __init__(self, wl, fl, clamp=True, symmetric=False):
        assert wl > 0, "invalid bits for word length: {}".format(wl)
        assert fl > 0, "invalid bits for fractional length: {}".format(fl)
        assert type(symmetric) == bool, "invalid type for clamping choice: {}".format(type(clamp))
        assert type(symmetric) == bool, "invalid type for symmetric: {}".format(type(symmetric))
        self.wl = wl
        self.fl = fl
        self.clamp = clamp
        self.symmetric = symmetric

    def __str__(self):
        return "FixedPoint Number with wl={:d}, fl={:d}".format(self.wl, self.fl)

class BlockFloatingPoint(Number):
    """
    Low-Precision Block Floating Point Format.
    Currently only supports treating the entire input tensor as a block.

    Args:
        - :attr: `wl` word length of the tensor
    """
    def __init__(self, wl):
        assert wl > 0, "invalid bits for word length:{}".format(wl)
        self.wl = wl

    def __str__(self):
        return "BlockFloatingPoint Number with wl={:d}".format(self.wl)

    def __repr__(self):
        return "BlockFloatingPoint Number with wl={:d}".format(self.wl)

class FloatingPoint(Number):
    """
    Low-Precision Floating Point Format

    Args:
        - :attr: `exp`: number of bits allocated for exponent
        - :attr: `man`: number of bits allocated for mantissa, referring to number of bits that are
                        supposed to be stored on hardware (not counting the virtual bits).
    """
    def __init__(self, exp, man):
        assert 8 >= exp > 0, "invalid bits for exponent:{}".format(exp)
        assert 23 >= man > 0, "invalid bits for mantissa:{}".format(man)
        self.exp = exp
        self.man = man

    def __str__(self):
        return "FloatingPoint Number with exponent={:d}, mantissa={:d}".format(self.exp, self.man)

    def __repr__(self):
        return "FloatingPoint Number with exponent={:d}, mantissa={:d}".format(self.exp, self.man)
