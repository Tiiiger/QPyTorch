__all__ = ['Number', 'FixedPoint', 'BlockFloatingPoint', 'FloatingPoint']

class Number():
    def __init__(self):
        pass

    def __str__(self):
        raise NotImplemented

class FixedPoint(Number):
    r"""
    Low-Precision Fixed Point Format. Defined similarly in
    *Deep Learning with Limited Numerical Precision* (https://arxiv.org/abs/1502.02551)

    The representable range is :math:`[-2^{wl-fl-1}, 2^{wl-fl-1}-2^{-fl}]`
    and a precision unit (smallest nonzero absolute value) is
    :math:`2^{-fl}`.
    Numbers outside of the representable range can be clamped
    (if `clamp` is true).
    We can also give up the smallest representable number to make the range
    symmetric, :math:`[-2^{wl-fl-1}^{-fl}, 2^{wl-fl-1}-2^{-fl}]`. (if `symmetric` is true).

    Define :math:`\lfloor x \rfloor` to be the largest representable number (multiples of :math:`2^{-fl}`) smaller than :math:`x`.
    For numbers within the representable range, fixed point quantizatio corresponds to

    .. math::

       NearestRound(x)
       =
       \Biggl \lbrace
       {
       \lfloor x \rfloor, \text{ if } \lfloor x \rfloor \leq x \leq \lfloor x \rfloor + 2^{-fl-1}
       \atop
        \lfloor x \rfloor + 2^{-fl}, \text{ if } \lfloor x \rfloor + 2^{-fl-1} < x \leq \lfloor x \rfloor + 2^{-fl}
       }

    or

    .. math::
       StochasticRound(x)
       =
       \Biggl \lbrace
       {
       \lfloor x \rfloor, \text{ with probabilty } 1 - \frac{x - \lfloor x \rfloor}{2^{-fl}}
       \atop
        \lfloor x \rfloor + 2^{-fl}, \text{ with probabilty } \frac{x - \lfloor x \rfloor}{2^{-fl}}
       }

    Args:
        - :attr: wl (int) : word length of each fixed point number
        - :attr: fl (int) : fractional length of each fixed point number
        - :attr: clamp (bool) : whether to clamp unrepresentable numbers
        - :attr: symmetric (bool) : whether to make the representable range symmetric
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
        return "FixedPoint (wl={:d}, fl={:d})".format(self.wl, self.fl)

    def __repr__(self):
        return "FixedPoint (wl={:d}, fl={:d})".format(self.wl, self.fl)

class FloatingPoint(Number):
    """
    Low-Precision Floating Point Format.

    We set the exponent bias to be :math:`2^{exp-1}`. In our simulation, we do
    not handle denormal/subnormal numbers and infinities/NaNs. For rounding
    mode, we apply *round to nearest even*.

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
        return "FloatingPoint (exponent={:d}, mantissa={:d})".format(self.exp, self.man)

    def __repr__(self):
        return "FloatingPoint (exponent={:d}, mantissa={:d})".format(self.exp, self.man)

class BlockFloatingPoint(Number):
    """
    Low-Precision Block Floating Point Format.

    BlockFloatingPoint shares an exponent across a block of numbers. The shared exponent is chosen from
    the largest magnitude in the block.

    Args:
        - :attr: `wl` word length of the tensor
        - :attr: `dim` block dimension to share exponent. (*, D, *) Tensor where
          D is at position `dim` will have D different exponents; use -1 if the
          entire tensor is treated as a single block (there is only 1 shared
          exponent).
    """
    def __init__(self, wl, dim=-1):
        assert wl > 0 and isinstance(wl, int), "invalid bits for word length:{}".format(wl)
        assert dim >= -1 and isinstance(dim, int), "invalid dimension"
        self.wl = wl
        self.dim = dim

    def __str__(self):
        return "BlockFloatingPoint (wl={:d}, dim={:d})".format(self.wl, self.dim)

    def __repr__(self):
        return "BlockFloatingPoint (wl={:d}, dim={:d})".format(self.wl, self.dim)
