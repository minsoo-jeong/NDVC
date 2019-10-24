class Period(object):
    def __init__(self, start=-1, end=-1):
        self.start = start
        self.end = end

        # if self.start < 0 and self.end < 0 or self.start > self.end:
        #    raise ValueError('value is more 0 or start >= end')

    def __repr__(self):
        return '{} - {}'.format(self.start, self.end)

    def __len__(self):
        return self.end - self.start

    def is_overlap(self, p):
        if not isinstance(p, Period):
            raise ValueError('p is not Period object')

        return not (p.start > self.end or p.end < self.start)

    def intersection(self, p):
        if not isinstance(p, Period):
            raise ValueError('p is not Period object')

        if not self.is_overlap(p): return False

        max_start = max(self.start, p.start)
        min_end = min(self.end, p.end)
        return Period(max_start, min_end)

    def union(self, p):
        if not isinstance(p, Period):
            raise ValueError('p is not Period object')

        if not self.is_overlap(p): return False

        min_start = min(self.start, p.start)
        max_end = max(self.end, p.end)

        return Period(min_start, max_end)

    def IOU(self, p):
        try:
            intersect = self.intersection(p)
            union = self.union(p)
            iou = len(intersect) / len(union)
        except:
            iou = 0

        return iou

    def isin(self, p):
        if not isinstance(p, Period):
            raise ValueError('p is not Period object')
        return (p.start <= self.start) and (self.end <= p.end)

    def is_replace(self, p):
        if not isinstance(p, Period):
            raise ValueError('p is not Period object')
        return (self.start <= p.start) and (p.end <= self.end)

    def __sub__(self, p):
        if not isinstance(p, Period):
            raise ValueError('p is not Period object')

        if self.is_overlap(p):
            max_start = max(self.start, p.start)
            min_end = min(self.end, p.end)
            intersect= Period(max_start, min_end)


if __name__ == '__main__':
    p=Period(0.3,1.5)
    b=Period(0.3,1.0)
    print(p,b,p.is_overlap(b))



    exit()
    l = [[1], [2], [3], [4], [5]]
    for n, i in enumerate(l):
        if i == [2]: l[2] = -1
    print(l)
    exit()
    a = Period(0, 8)
    b = Period(0, 52)
    print(a.is_overlap(b))

    print(a.intersection(b))
    print(a.union(b))
    print(a.IOU(b))

    print(Period.IOU(a, b))
