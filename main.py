import numpy
import matplotlib.pyplot as plt
from copy import deepcopy
from skimage.io import imread, imshow
from PIL import Image
import numpy


class HilbertPoint:

    def __init__(self, n, x, y):

        self.n = n
        self.x = x
        self.y = y

    def __repr__(self):
        return "X: {} | Y: {}".format(self.x, self.y)


DEPTH = 5
MIN_X = 0 * DEPTH
MAX_X = 2 * DEPTH
MIN_Y = 0 * DEPTH
MAX_Y = 2 * DEPTH
ROUNDING_CONSTANT = 4  # digits


def get_fill_points(start, end):
    count = 2
    pts = []
    if round(start.x, ROUNDING_CONSTANT) == round(end.x, ROUNDING_CONSTANT):
        for n in numpy.linspace(start.y, end.y, num=count):
            pts.append(HilbertPoint(n, start.x, n))
    elif round(start.y, ROUNDING_CONSTANT) == round(end.y, ROUNDING_CONSTANT):
        for n in numpy.linspace(start.x, end.x, num=count):
            pts.append(HilbertPoint(n, n, start.y))
    else:
        raise RuntimeError(str(start) + str(end))

    return pts


class HilbertBox:

    def __init__(self, point_list):
        # copy the contents of the given points so we can transform them
        self.point_list = PointList(point_list.start_x, point_list.start_y)
        self.point_list.contents = point_list.contents

    def connect(self, other_box):
        joiner = get_fill_points(self.end, other_box.start)
        self.point_list.extend(joiner)
        self.point_list.extend(other_box.point_list.contents)

    @property
    def start(self):
        return self.point_list.start

    @property
    def end(self):
        return self.point_list.end

    def draw(self):
        plt.scatter(self.point_list.x_list, self.point_list.y_list)
        gca = plt.gca()
        gca.set_xlim([MIN_X, MAX_X])
        gca.set_ylim([MIN_X, MAX_Y])
        plt.show()


class BLHilbertBox(HilbertBox):

    def __init__(self, point_list):
        super().__init__(point_list)
        self.point_list.rotate(False)
        self.point_list.reverse()


class TLHilbertBox(HilbertBox):

    def __init__(self, point_list):
        super().__init__(point_list)
        self.point_list.offset_transform(-point_list.size, False)


class TRHilbertBox(HilbertBox):

    def __init__(self, point_list):
        super().__init__(point_list)
        self.point_list.offset_transform(-point_list.size, False)
        self.point_list.offset_transform(-point_list.size, True)


class BRHilbertBox(HilbertBox):

    def __init__(self, point_list):
        super().__init__(point_list)
        self.point_list.offset_transform(-point_list.size, True)
        self.point_list.rotate(True)
        self.point_list.reverse()


class PointList:

    EDGE_PADDING = 1

    def __init__(self, start_x, start_y):
        self._pts = []
        self.start_x = start_x
        self.start_y = start_y
        self.start = None
        self.end = None

    def build_staple(self):
        self.start = HilbertPoint(0, self.start_x, self.start_y)
        top_left = HilbertPoint(1, self.start_x, self.start_y + self.EDGE_PADDING)
        top_right = HilbertPoint(2, self.start_x + self.EDGE_PADDING, self.start_y + self.EDGE_PADDING)
        self.end = HilbertPoint(3, self.start_x + self.EDGE_PADDING, self.start_y)

        fill = get_fill_points(start=self.start, end=top_left)

        self._pts.extend(fill)

        fill = get_fill_points(start=top_left, end=top_right)

        self._pts.extend(fill)

        fill = get_fill_points(start=top_right, end=self.end)

        self._pts.extend(fill)

    @property
    def contents(self):
        pass

    @contents.getter
    def contents(self):
        return deepcopy(self._pts)

    @contents.setter
    def contents(self, pts):
        self.start = pts[0]
        self.end = pts[-1]
        self._pts = pts

    def extend(self, extend_list):
        self._pts.extend(extend_list)
        self.end = extend_list[-1]

    def offset_transform(self, offset, x):
        for pt in self._pts:
            if x:
                pt.x = pt.x - offset
            else:
                pt.y = pt.y - offset

    def reverse(self):
        self._pts.reverse()
        start = self.start
        self.start = self.end
        self.end = start

    def center(self):
        width = max(self.x_list) - min(self.x_list)
        height = max(self.y_list) - min(self.y_list)
        offset_x = min(self.x_list) + width/2
        offset_y = min(self.y_list) + height/2
        self.offset_transform(offset_x, True)
        self.offset_transform(offset_y, False)
        return offset_x, offset_y

    def rotate_about_center(self, cc):
        if cc:
            for pt in self._pts:
                old_x = pt.x
                pt.x = -pt.y
                pt.y = old_x
        else:
            for pt in self._pts:
                old_x = pt.x
                pt.x = pt.y
                pt.y = -old_x

    def decenter(self, offset_x, offset_y):
        self.offset_transform(-offset_x, True)
        self.offset_transform(-offset_y, False)

    def rotate(self, cc):
        offset_x, offset_y = self.center()
        self.rotate_about_center(cc)
        self.decenter(offset_x, offset_y)

    def flip_about_center(self):
        for pt in self._pts:
            pt.x = -pt.x

    def flip(self):
        offset_x, offset_y = self.center()
        self.flip_about_center()
        self.decenter(offset_x, offset_y)

    def __repr__(self):
        return str([pt for pt in self._pts])

    @property
    def x_list(self):
        return [pt.x for pt in self._pts]

    @property
    def y_list(self):
        return [pt.y for pt in self._pts]

    @property
    def n_s(self):
        return [pt.n for pt in self._pts]

    @property
    def size(self):
        width = max(self.x_list) - min(self.x_list)
        height = max(self.y_list) - min(self.y_list)

        if width == 0.0:
            size = height
        else:
            size = width

        # correct for padding
        size += self.EDGE_PADDING
        return size


def tile_boxes(point_list):
    bl_hb = BLHilbertBox(point_list)
    tl_hb = TLHilbertBox(point_list)
    br_hb = BRHilbertBox(point_list)
    tr_hb = TRHilbertBox(point_list)

    bl_hb.connect(tl_hb)
    tr_hb.connect(br_hb)
    bl_hb.connect(tr_hb)

    return bl_hb.point_list


def get_points(start_x, start_y, recursive_depth):
    point_list = PointList(start_x, start_y)
    point_list.build_staple()

    for i in range(0, recursive_depth):
        point_list = tile_boxes(point_list)

    return point_list


def call_hilbert():
    point_list = get_points(start_x=0, start_y=0, recursive_depth=DEPTH)
    plt.scatter(point_list.x_list, point_list.y_list)
    gca = plt.gca()
    gca.set_xlim([MIN_X, point_list.size])
    gca.set_ylim([MIN_X, point_list.size])
    #plt.show()
    return point_list


def get_image():
    im = Image.open('assets/lion.jpg', 'r')
    data = im.getdata()
    return data


def sample_hilbert(image, points):
    width, height = image.size
    image_list = list(image)
    reds = []

    pts = points.contents

    for i, pt in enumerate(pts):
        float_index = pt.y*width + pt.x
        number = round(float_index)
        val = image_list[number]
        reds.append(val[0])
    print(len(reds))

    chunks = [reds[x:x + width] for x in range(0, len(reds), width)]
    #print(chunks)

    chunks = chunks[:-1]

    return chunks


def show_image(reds):
    array = numpy.array(reds)
    new_image = Image.fromarray(reds)
    new_image.save('reds.png')


def call_all():
    image = get_image()
    points = call_hilbert()
    reds = sample_hilbert(image, points)
    show_image(reds)


if __name__ == "__main__":
    call_all()