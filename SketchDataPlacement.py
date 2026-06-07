import hashlib
import array


class CountMinSketch(object):
    
    def __init__(self, m, d):
        """ `m` is the size of the hash tables, larger implies smaller
        overestimation. `d` the amount of hash tables, larger implies lower
        probability of overestimation.
        """
        if not m or not d:
            raise ValueError("Table size (m) and amount of hash functions (d)"
                             " must be non-zero")
        self.m = m
        self.d = d
        self.n = 0
        self.tables = []
        for _ in range(d):
            table = array.array("l", (0 for _ in range(m)))
            self.tables.append(table)

    def _hash(self, x):
        md5 = hashlib.sha256(x.encode('utf-8'))
        for i in range(self.d):
            md5.update(str(i).encode('utf-8'))
            yield int(md5.hexdigest(), 16) % self.m

    def add(self, x, value=1):
        """
        Count element `x` as if had appeared `value` times.
        By default `value=1` so:
            sketch.add(x)
        Effectively counts `x` as occurring once.
        """
        self.n += value
        for table, i in zip(self.tables, self._hash(x)):
            table[i] += value

    def query(self, x):
        """
        Return an estimation of the amount of times `x` has ocurred.
        The returned value always overestimates the real value.
        """
        return min(table[i] for table, i in zip(self.tables, self._hash(x)))

    def getitem(self, x):
        """
        A convenience method to call `query`.
        """
        return self.query(x)

    def len(self):
        """
        The amount of things counted. Takes into account that the `value`
        argument of `add` might be different from 1.
        """
        return self.n
