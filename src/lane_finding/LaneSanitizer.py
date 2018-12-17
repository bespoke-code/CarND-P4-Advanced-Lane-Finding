import numpy as np


class LaneSanitizer:
    """
    Holds a queue of lane line polynomials. Can calculate average new line polynomials.
    """
    weights = [0.1, 0.1, 0.15, 0.3, 0.35]  # used for weighed average calculation
    ym_per_pix = 30. / 720.  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 640.
    ploty = None

    def __init__(self, width, height):
        self.line_queue = []
        self.capacity = 5
        self.curr_count = 0  # left = index 0, right = index
        self.ploty = np.linspace(0, height - 1, width)
        #print('ploty dimensions: ', self.ploty.shape)

    def calculate_new_line(self):
        #return np.average(self.line_queue, axis=0, weights=self.weights[-1*self.curr_count:])
        return np.mean(self.line_queue, axis=0)
        #return self.line_queue[-1]  #, weights=self.weights[-1*self.curr_count:])

    def _slope(self, new_line):
        """
        Calculates approximate slope of a lane line.
        :param new_line: Quadratic coefficients for the line.
        :return: A slope angle value in [rad]
        """
        y_eval = np.min(self.ploty)
        line_top = new_line[0] * y_eval**2 + new_line[1] * y_eval + new_line[2]
        y_eval = np.max(self.ploty)
        line_bottom = new_line[0] * y_eval**2 + new_line[1] * y_eval + new_line[2]
        return np.arctan((line_bottom - line_top)/(y_eval - np.min(self.ploty)))

    def is_similar(self, new_line):
        #print(np.round(self._slope(new_line) - self._slope(self.calculate_new_line()), 3))
        #return np.abs(self._slope(new_line) - self._slope(self.calculate_new_line())) < 0.05
        return np.abs(self._slope(new_line) - self._slope(self.calculate_new_line())) < 0.3

    def add(self, new_line_polyfit):
        """
        Checks a line and adds it to the list if the sanity check is good.
        :param new_line_polyfit: the new line's quadratic equation.
        :return: True if the line is added to the queue, or false
        if it isn't good and is therefore not added to the queue.
        """
        if self.curr_count == 0:
            self.line_queue.append(new_line_polyfit)
            self.curr_count += 1
            return True

        if self.is_similar(new_line_polyfit):
            if self.curr_count == self.capacity:
                self.line_queue.pop(0)
                self.curr_count -= 1
            self.line_queue.append(new_line_polyfit)
            self.curr_count += 1
            return True
        else:
            return False

    def get_last(self, last_line_ok=False):
        if last_line_ok:
            poly = self.line_queue[-1]
            fitx = poly[0] * self.ploty**2 + poly[1] * self.ploty + poly[2]
        else:
            poly = self.calculate_new_line()
            fitx = poly[0] * self.ploty ** 2 + poly[1] * self.ploty + poly[2]
        return poly, fitx