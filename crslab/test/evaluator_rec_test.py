import unittest
from crslab.evaluator.rec import RecEvaluator


class HRTestCase():
    def __init__(self):
        self.evaluator = RecEvaluator()

    def hr_test(self):
        """
        1-10 items, 1-5 are correct, hr@5
        """
        pred = [5, 4, 3, 6, 7, 8, 9, 1, 2]
        real = [1, 2, 3, 4, 5]
        self.evaluator.rec_evaluate(pred, real)
        self.evaluator.report()


hr = HRTestCase()
hr.hr_test()