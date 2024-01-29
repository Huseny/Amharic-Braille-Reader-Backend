class BrailleChar:
    def __init__(self, box, label):
        self.original_box = box # box found by NN
        self.x = (box[0] + box[2])/2 # original x of last char
        self.y = (box[1] + box[3])/2 # original y of last char
        self.w = (box[2]-box[0]) # original w
        self.h = (box[3]-box[1]) # original h
        self.approximation = None # approximation (err,a,b,w,h) on interval ending on this
        self.refined_box = box # refined box
        self.label = label
        self.spaces_before = 0


class Line:
    STEP_TO_W = 1.25
    LINE_THR = 0.6
    AVG_PERIOD = 5
    AVG_APPROX_DIST = 3

    def __init__(self, box, label):
        self.chars: list[BrailleChar] = []
        new_char: BrailleChar = BrailleChar(box, label)
        self.chars.append(new_char)
        self.x = new_char.x
        self.y = new_char.y
        self.h = new_char.h
        self.slip = 0
        self.has_space_before = False

    def check_and_append(self, box, label):
        x = (box[0] + box[2])/2
        y = (box[1] + box[3])/2
        if abs(self.y + self.slip * (x-self.x) -y) < self.h*self.LINE_THR:
            new_char = BrailleChar(box, label)
            self.chars.append(new_char)
            calc_chars = self.chars[-self.AVG_PERIOD:]
            new_char.approximation = self._calc_approximation(calc_chars)
            if new_char.approximation is None:
                self.x = new_char.x
                self.y = new_char.y
                self.h = new_char.h
            else:
                err, a, b, w, h = new_char.approximation
                self.x = new_char.x
                self.y = self.x * a + b
                self.h = h
                self.slip = a
            return True
        return False

    def _calc_approximation(self, calc_chars):
        if len(calc_chars) <= self.AVG_APPROX_DIST:
            return None
        best_pair = [1e99, None, None, None, None, ] # err,i,j,k,a,b
        for i in range(len(calc_chars)-self.AVG_APPROX_DIST):
            for j in range(i+self.AVG_APPROX_DIST, len(calc_chars)):
                a = (calc_chars[j].y - calc_chars[i].y) / (calc_chars[j].x - calc_chars[i].x)
                b = (calc_chars[j].x * calc_chars[i].y - calc_chars[i].x * calc_chars[j].y) / (calc_chars[j].x - calc_chars[i].x)
                ks = [k for k in range(len(calc_chars)) if k!=i and k!= j]
                errors = [abs(calc_chars[k].x * a + b - calc_chars[k].y) for k in ks]
                min_err = min(errors)
                if min_err < best_pair[0]:
                    idx_if_min = min(range(len(errors)), key=errors.__getitem__)
                    best_pair = min_err, i, j, ks[idx_if_min], a, b
        err, i, j, k, a, b = best_pair
        w = (calc_chars[i].w + calc_chars[j].w + calc_chars[k].w) / 3
        h = (calc_chars[i].h + calc_chars[j].h + calc_chars[k].h) / 3
        return err, a, b, w, h

    def refine(self):
        for i in range(len(self.chars)):
            curr_char = self.chars[i]
            ref_chars = [ch for ch in  self.chars[i:i+self.AVG_PERIOD] if ch.approximation is not None]
            if ref_chars:
                best_char = min(ref_chars, key = lambda ch: ch.approximation[0])
                err, a, b, w, h = best_char.approximation
                expected_x = curr_char.x
                expected_y = expected_x * a + b
                curr_char.refined_box = [expected_x-w/2, expected_y-h/2, expected_x+w/2, expected_y+h/2]
            if i > 0:
                step = (curr_char.refined_box[2] - curr_char.refined_box[0]) * self.STEP_TO_W
                prev_char = self.chars[i-1]
                curr_char.spaces_before = max(0, round(0.5 * ((curr_char.refined_box[0] + curr_char.refined_box[2]) - (prev_char.refined_box[0] + prev_char.refined_box[2])) / step) - 1)