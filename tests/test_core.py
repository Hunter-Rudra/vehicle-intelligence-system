import unittest
import numpy as np
from vehicle_intelligence.preprocessing import to_grayscale, gaussian_kernel
from vehicle_intelligence.components import connected_components
from vehicle_intelligence.background import RunningAverageBackground
from vehicle_intelligence.counting import LineCounter

class CoreTests(unittest.TestCase):
    def test_grayscale_shape(self):
        self.assertEqual(to_grayscale(np.zeros((4,5,3))).shape,(4,5))
    def test_kernel_normalized(self):
        self.assertAlmostEqual(float(gaussian_kernel().sum()),1.0,places=5)
    def test_component(self):
        m=np.zeros((10,10),bool); m[2:5,3:7]=True
        c=connected_components(m,1)
        self.assertEqual(c[0]["area"],12)
        self.assertEqual(c[0]["bbox"],(3,2,4,3))
    def test_first_background_frame(self):
        b=RunningAverageBackground()
        self.assertFalse(b.apply(np.zeros((5,5))).any())
    def test_count_once(self):
        c=LineCounter(10)
        c.update({7:(2,2)})
        self.assertEqual(c.update({7:(2,15)}),1)
        c.update({7:(2,2)})
        self.assertEqual(c.count,1)

if __name__ == "__main__":
    unittest.main()
