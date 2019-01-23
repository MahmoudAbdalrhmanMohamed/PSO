import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation


class PSO:
    def __init__(self, c1=1.0, c2=1.0, dimension=2, pop_size=2, max_iter=10):
        self.dimension = dimension
        self.pop_size = pop_size
        self.max_iter = max_iter

        # Accelerations constants
        self.c1 = c1  # cognitive constant (p_best)
        self.c2 = c2  # social constant (g_best)
        self.w = 0.9  # Inertia

        self.cost_func = None  # Cost function
        self.X = None  # Positions
        self.V = None  # Velocities
        self.p_best = None  # Particles best positions so far
        self.g_best = None  # Global best position

        # Keeps a list of g_best at each iteration to plot the evolution
        self.evolution = []
        # Keep the list of averages on each iteration to plot at the end
        self.avg = []

        # Holds all positions to generate an scatter plot
        self.scatter = []  # np.zeros((self.pop_size, self.dimension, self.max_iter))

    def random_init(self):
        self.X = np.random.rand(self.pop_size, self.dimension)
        self.V = np.random.rand(self.pop_size, self.dimension)
        # At first p_best is equal to first positions
        self.p_best = np.copy(self.X)
        # At first we consider that the X_1 is the global best
        self.g_best = self.X[0]

    def update_p_best(self):
        for index, values in enumerate(self.X):
            cost = self.cost_func(values)
            if cost < self.cost_func(self.p_best[index]):
                self.p_best[index] = values

    def update_g_best(self):
        for index, values in enumerate(self.p_best):
            cost = self.cost_func(values)
            if cost < self.cost_func(self.g_best):
                self.g_best = values

    def update_velocity(self):
        r1 = np.random.random_sample()
        r2 = np.random.random_sample()
        self.V = (self.w * self.V +
                  self.c1 * r1 * (self.p_best - self.X) +
                  self.c2 * r2 * (self.g_best - self.X))

    def update_positions(self):
        self.X = self.X + self.V

    def update_w(self, i):
        self.w = 0.9 - ((0.9 - 0.2) / self.max_iter) * i

    def cal_avg(self):
        sum_of_costs = 0
        for values in self.X:
            sum_of_costs += self.cost_func(values)
        return sum_of_costs / self.pop_size

    def start(self):
        # Save first position (animated scatter plot)
        self.scatter.append(np.copy(self.X))
        i = 0
        while i < self.max_iter:
            self.update_p_best()
            self.update_g_best()
            self.update_velocity()
            self.update_positions()
            self.update_w(i)

            self.scatter.append(np.copy(self.X))
            # Add the g_best value (cost) at this iteration (to plot)
            self.evolution.append(self.cost_func(self.g_best))
            self.avg.append(self.cal_avg())
            # update velocity
            # update positions
            i += 1


def rosenbrock(args):
    x, y = args
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def dixon_price(x):
    return (x[0] - 1) ** 2 + sum((i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2 for i in range(1, len(x)))


# - - - - - - -  - - - - - - -  - - - - - - -  - - - - - - -
#                   RUNNING THE PSO
# - - - - - - -  - - - - - - -  - - - - - - -  - - - - - - -

pso = PSO(c1=2, c2=1.5, dimension=2, pop_size=5, max_iter=240)
pso.cost_func = rosenbrock
pso.random_init()
pso.start()

# - - - - - - - - Printing the results - - - - - - - - - - -

print('Global best position:', pso.g_best)
print('Cost function in global best:', "{:.6f}".format(float(pso.evolution[-1])))

# - - - - - - - - - Plotting G-BEST & AVG - - - - - - - - - -

# show evolution and averages in a plot
plt.figure(figsize=(12, 16))

evolution = plt.subplot(2, 1, 1)
evolution.set_title('G-Best in each iteration')
evolution.plot(pso.evolution, color='darkviolet')
evolution.set_ylim(-0.01, 0.02)
# Put a point on where global best is
evolution.plot([pso.evolution.index(pso.evolution[-1])], [pso.evolution[-1]], '.', color='red')

avg = plt.subplot(2, 1, 2)
avg.set_title('Average of population in each iteration')
#avg.set_ylim(-0.01, 1)
avg.plot(pso.avg, color='deeppink')
plt.show()

# ---------------------------- Add animation -------------------------
if pso.dimension != 2:
    raise Exception("Only able to animate 2 dimensional functions.")

fig = plt.figure()
x, y = np.split(pso.scatter[0], [-1], axis=1)
a, = plt.plot(x, y, "o", color="darkviolet")
plt.xlim(right=1.2, left=0)
plt.ylim(top=1.2, bottom=0)


def update():
    yield from pso.scatter


def animate(data):
    x, y = np.split(data, [-1], axis=1)
    a.set_xdata(x)
    a.set_ydata(y)

    return a,


anim = animation.FuncAnimation(fig, animate, update(), interval=55, repeat=False, blit=True)

plt.show()
