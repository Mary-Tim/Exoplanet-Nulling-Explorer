import torch
import torch.nn as nn

class CubicSplineInterpolation(nn.Module):
    def __init__(self, points):
        super(CubicSplineInterpolation, self).__init__()
        self.points = points
        self.n = len(points)
        self.x = torch.tensor(points, dtype=torch.float32)
        self.y = torch.tensor([i for i in range(self.n)], dtype=torch.float32)
        self.coefficients = self.compute_coefficients()

    def compute_coefficients(self):
        # 计算三次样条插值的系数
        h = self.x[1:] - self.x[:-1]
        A = torch.zeros((self.n - 1, self.n - 1))
        b = torch.zeros((self.n - 1, 1))

        A[:, 0] = h[:-1] / 3
        A[:, 1] = (h[1:] + 2 * h[:-1]) / 3
        A[:, 2] = h[1:] / 6
        b[:, 0] = self.y[1:] - self.y[:-1]

        return torch.linalg.solve(A, b)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        idx = torch.searchsorted(self.x, x)
        idx = torch.clamp(idx, 0, self.n - 1)

        x0 = self.x[idx]
        x1 = self.x[idx + 1]
        y0 = self.y[idx]
        y1 = self.y[idx + 1]
        c0 = self.coefficients[idx]
        c1 = self.coefficients[idx + 1]

        dx = x - x0
        dx1 = x1 - x

        y = y0 + (y1 - y0) * ((c0[0] * dx + c0[1]) * dx + c0[2]) * dx
        y += (y1 - y0) * ((c1[0] * dx1 + c1[1]) * dx1 + c1[2]) * dx1

        return y

def test_cubic_spline_interpolation():
    points = torch.tensor([0, 1, 2, 3])
    interp = CubicSplineInterpolation(points)

    # Test points
    test_points = torch.tensor([0, 0.5, 1, 1.5, 2, 2.5, 3])
    y = interp(test_points)
    print("Test points: ", test_points)
    print("Interpolated values: ", y)

    # Test gradient
    y_grad = interp.compute_gradient(test_points)
    print("Gradients: ", y_grad)

# Run the test
test_cubic_spline_interpolation()