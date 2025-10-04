# ASTR 513 Homework Set #3

Welcome to the repository for **Homework Set #3** in ASTR 513.
This homework set is worth **5 points** and is designed to test your
understanding of topics that we covered in the first three classes.
The submission cutoff time is at **Friday Oct 10th, 11:59pm** Arizona
time.
Use this GitHub Classroom Link:
https://classroom.github.com/a/7xV0a93o
to accept it.


## Structure and Grading

This homework set consists of **five parts**, each contributing
equally to your overall grade.

* Each part will be graded using `pytest`, an automated testing
  framework.
* The correctness and the documentations/comments of your solutions
  accounts for 1 point per part (5 points in total).
* You can run `pytest` in GitHub Codespaces or your local development
  environment to verify your solutions before submission.

By following the interface and prototypes provided in each of the
templates, you can ensure compatibility with the autograding system
and maximize your score.


## Parts

### Part 1: Package Setup (1 point)

* Objective:
  Turn your repository into a pip-installable Python package.
* Details:
  * As in HW1 and HW2, create a pyproject.toml and add a LICENSE file.
    This allows you to install your homework with:
    ```bash
    pip install -e .
    ```
  * Organize your code in a `hw3/` directory with a clear module
    layout.
  * Include documentation (e.g., update this `README.md`) so that your
    package is self-contained and easy to use.

### Part 2: RK4 Integrator (1 point)

* Objective:
  Implement a reusable 4th-order Runge-Kutta (RK4) integrator for
  systems of ODEs.
* Details:
  * Function prototype:
    ```python
    def RK4(
        f,   # right-hand side function: dx/dt = f(x)
        x,   # initial state vector x(t0)
        t,   # initial time t0
        dt,  # timestep size
        n,   # number of steps to evolve
    ):
        ...
    ```
  * Return arrays of states (`X`) and times (`T`) for all integration
    steps.  I.e., "history" in our lecture notes.
  * Demonstrate correctness by integrating a simple harmonic
    oscillator and comparing with the analytic solution.
    See `demo.ipynb`.

### Part 3: Euler-Lagrange Equation via Autodiff (1 point)

* Objective:
  Use JAX automatic differentiation to derive Euler-Lagrange equations
  from an input Lagrangian.
* Details:
  * Function prototype:
    ```python
    def ELrhs(L):
        ...
    ```
    where `L(x, v)` is a Python function defining the Lagrangian, and
    `x`, `v` are generalized coordinates and velocities.
  * `ELrhs(L)` should return a function `def rhs(xv): ...` where where
    `xv` is the concatenated state `[x, v]`, and the output is
    `[v, a]` (velocities and "accelerations").
    This is possible because python supports functional programming.
    Functions are "first-class citizens" in python.
  * Use `jax.grad()` and `jax.jacfwd()` to compute the necessary
    derivatives for the Euler-Lagrange equation:

    $$\frac{d}{dt}\frac{\partial L}{\partial\dot{q}} - \frac{\partial L}{\partial q} = 0.$$

    You may need to read
    [JAX's documentation](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html)
    to make it work.
  * Verify correctness on a single harmonic oscillator and confirm
    that the derived ODE matches the known analytic form.
    See `demo.ipynb`.

### Part 4: Double Pendulum Lagrangian (1 point)

* Objective:
  Implement and solve the Lagrangian of a doublea pendulum problem.
* Details:
  * Define the Lagrangian of a two-dimensional double pendulum
    problem
    ```python
    def L(x, v):
        ...
    ```
    where `x = [theta1, theta2]` and `v = [omega1, omega2]`.
  * The notations $\theta_1$ and $\theta_2$ follow
    [wikipedia](https://en.wikipedia.org/wiki/Double_pendulum)
    so we simply implement

    $$L = \frac{1}{6} ml^2 \Big[\dot\theta_2^2
                             + 4\dot\theta_1^2
                             + 3\dot\theta_1\dot\theta_2\cos(\theta_1-\theta_2)\Big]
        + \frac{1}{2} mgl  \Big[3\cos\theta_1 + \cos\theta_2\Big].$$
    Please check the above equation carefully as it may contain typos.
  * This Lagrangian will be passed to `ELrhs()` (Part 3) to generate
    equations of motion automatically.

### Part 5: Create a Movie (1 point)

* Objective:
  Put everything together to solve the double pendulum problem.
* Details:
  * Implement:
    ```python
    def solve(
        theta1, # initial angle for pendulum 1
        theta2, # initial angle for pendulum 2
        omega1, # initial angular velocity for pendulum 1
        omega2, # initial angular velocity for pendulum 2
        t,      # total simulation time
        dt,     # time step size
    ):
        ...
    ```
  * Use `L()` from Part 4 as the Lagrangian.
  * Use `ELrhs()` from Part 3 to get the right-hand side.
  * Use `RK4()` from Part 2 to integrate the system.
  * Return the tuple `X, V, T`, where:
    * `X`: array of angles over time with shape `(n+1, 2)`.
    * `V`: array of angular velocities with shape `(n+1, 2)`.
    * `T`: array of times with shape `(n+1,)`.
  * See `demo.ipynb`.


## Submission Guidelines

1. Create a new repostiory based on this template by using the
   provided GitHub Classroom Link and work on the different parts
   locally on your computer or in GitHub Codespaces.
2. Use `pytest` to test your solutions before submission.
   Ensure that all tests pass to maximize your automated evaluation
   score.
3. **Submit your completed homework set by pushing your changes to the
   repository before the deadline.**

**Note**:
**The submission deadline is a strict cutoff.
No submission will be accepted after the deadline.**
Manage your time effectively and plan ahead to ensure your work is
submitted on time.
Frequent and incremental `git commit` and `git push` are recommended
to ensure your latest work are seen.


## Additional Notes

* Collaboration:
  You are encouraged to discuss ideas with your peers, but your
  submission must reflect your own work and understanding.
* AI:
  When using AI, you must follow the course policy.
  The bottomline is that you may use AI as a collaborator to help you
  learn, but the code must reflect your own work and understanding.
* Help and Resources:
  If you encounter any difficulties, please refer to the course
  materials, consult your instructor, or seek help during office
  hours.
* Deadlines:
  Be mindful of the submission deadline, as late submissions will not
  be accepted.

Good luck!
