from functools import cmp_to_key
from ..models.line import Line

def _get_compareble_y(line1: Line, line2: Line):
    line1, line2, sign = (line1, line2, 1) if line1.length > line2.length else (line2, line1, -1)
    if line2.chars[0].x > line1.middle_x:
        y1 = line1.chars[-1].y + line1.mean_slip * (line2.chars[0].x - line1.chars[-1].x)
    else:
        y1 = line1.chars[0].y + line1.mean_slip * (line2.chars[0].x - line1.chars[0].x)
    if sign > 0:
        return y1, line2.chars[0].y
    else:
        return line2.chars[0].y, y1


def _sort_lines(lines):

    def _cmp_lines(line1, line2):
        y1, y2 = _get_compareble_y(line1, line2)
        return 1 if y1 > y2 else -1

    for ln in lines:
        ln.length = ln.chars[-1].x - ln.chars[0].x
        ln.middle_x = 0.5*(ln.chars[-1].x + ln.chars[0].x)
        if ln.length > 0:
            ln.mean_slip = (ln.chars[-1].y - ln.chars[0].y)/ln.length
        else:
            ln.mean_slip = 0
    return sorted(lines, key = cmp_to_key(_cmp_lines))



def _filter_lonely_rects_for_lines(lines):
    allowed_lonely = {} # lt.label010_to_int('111000'), lt.label010_to_int('000111'), lt.label010_to_int('111111')
    filtered_chars = []
    for ln in lines:
        while len(ln.chars) and (ln.chars[0].label not in allowed_lonely and len(ln.chars)>1 and ln.chars[1].spaces_before > 1 or len(ln.chars) == 1):
            filtered_chars.append(ln.chars[0])
            ln.chars = ln.chars[1:]
            if len(ln.chars):
                ln.chars[0].spaces_before = 0
        while len(ln.chars) and (ln.chars[-1].label not in allowed_lonely and len(ln.chars)>1 and ln.chars[-1].spaces_before > 1 or len(ln.chars) == 1):
            filtered_chars.append(ln.chars[-1])
            ln.chars = ln.chars[:-1]
    return [ln for ln in lines if len(ln.chars)], filtered_chars


def boxes_to_lines(boxes, labels, filter_lonely = True):
    '''
    :param boxes: list of (left, tor, right, bottom)
    :return: text: list of strings
    '''
    VERTICAL_SPACING_THR = 2.3
    boxes = list(zip(boxes, labels))
    lines = []
    boxes = sorted(boxes, key=lambda b: b[0][0])

    for b in boxes:
        found_line = None
        for ln in lines:
            if ln.check_and_append(box=b[0], label=b[1]):
                # to handle seldom cases when one char can be related to several lines mostly because of errorneous outlined symbols
                if (found_line and (found_line.chars[-1].x - found_line.chars[-2].x) < (ln.chars[-1].x - ln.chars[-2].x)):
                    ln.chars.pop()
                else:
                    if found_line:
                        found_line.chars.pop()
                    found_line = ln
        if found_line is None:
            lines.append(Line(box=b[0], label=b[1]))

    lines = _sort_lines(lines)
    interpret_line_mode = None
    prev_line = None
    for ln in lines:
        ln.refine()

        if prev_line is not None:
            prev_y, y = _get_compareble_y(prev_line, ln)
            if (y - prev_y) > VERTICAL_SPACING_THR * ln.h:
                ln.has_space_before = True
        prev_line = ln

    if filter_lonely:
        lines, _ = _filter_lonely_rects_for_lines(lines)
    return lines