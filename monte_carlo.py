import random
import matplotlib.pyplot as plt

def estimate_pi(num_samples):
    inside_circle = 0

    for _ in range(num_samples):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        distance = x**2 + y**2

        if distance <= 1:
            inside_circle += 1

    return (inside_circle / num_samples) * 4

def plot_points_in_circle(num_samples):
    inside_x = []
    inside_y = []
    outside_x = []
    outside_y = []

    for _ in range(num_samples):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        distance = x**2 + y**2

        if distance <= 1:
            inside_x.append(x)
            inside_y.append(y)
        else:
            outside_x.append(x)
            outside_y.append(y)

    plt.scatter(inside_x, inside_y, color='blue', marker='.')
    plt.scatter(outside_x, outside_y, color='red', marker='.')
    plt.show()
num_samples = 100000
estimated_pi = estimate_pi(num_samples)
print(f"Estimated value of pi using {num_samples} samples: {estimated_pi}")

