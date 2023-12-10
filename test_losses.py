import unittest
import numpy as np
from losses import Loss, MSE

class LossTest(unittest.TestCase):
    def test_forward(self):
        loss = MSE(batch_size=4)
        y_pred = np.array([0.5, 0.7, 0.2, 0])
        y_true = np.array([1, 0, 1, 0])
        result = loss.forward(y_pred, y_true)
        self.assertEqual(result, 0.34500000000000003)  # Replace expected_value with the expected result

    def test_backward(self):
        loss = MSE(batch_size=4)
        y_pred = np.array([0.5, 0.7, 0.2, 0])
        y_true = np.array([1, 0, 1, 0])
        result = loss.backward(y_pred, y_true)
        # compare the two array element by element
        comp = result.all() == np.array([-1, 1.4, -1.6, 0]).all()
        self.assertEqual(comp, True)  # Replace expected_value with the expected result

    def test_check_shape(self):
        loss = Loss(batch_size=32)
        y_pred = np.array([0.5, 0.7, 0.2, 0])
        y_true = np.array([1, 0, 1])
        with self.assertRaises(ValueError):
            loss.check_shape(y_pred, y_true)

if __name__ == '__main__':
    unittest.main()