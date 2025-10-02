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
  * Include documentation (e.g., updatre this `README.md`) so that
    your package is self-contained and easy to use.

### Part 2: RK4 Integrator (1 point)

* Objective:
  Implement a reusable 4th-order Runge–Kutta (RK4) integrator for
  systems of ODEs.
* Details:
  * Function prototype:
    ```python
    def RK4(
        f,   # right-hand side function: dx/dt = f(t, x)
        x0,  # initial state vector x(t0)
        t0,  # initial time t0
        dt,  # timestep size
        nt,  # number of steps to evolve
    ):
    ...
    ```
  * Return arrays of times and states for all integration steps.
  * Demonstrate correctness by integrating a simple harmonic
    oscillator and comparing with the analytic solution.
    (You may do it in a `demo.ipynb` and submit it together with this
    package.)

### Part 3: Euler-Lagrange Equation via Autodiff (1 point)

* Objective:
  Use JAX automatic differentiation to derive Euler–Lagrange equations
  from an input Lagrangian.
* Details:
  * Function prototype:
    ```python
    def EL(L, q, qdot):
        ...
    ```
    where `L(q, qdot)` is a Python function defining the Lagrangian,
    and `q`, `qdot` are generalized coordinates and velocities.

  * Use `jax.grad()` to compute the necessary derivatives for the
    Euler-Lagrange equation:
    \begin{align}
      \frac{d}{dt}\frac{\partial L}{\partial\dot{q}} - \frac{\partial L}{\partial q} = 0
    \end{align}
  * Verify correctness on a single harmonic oscillator and confirm
    that the derived ODE matches the known analytic form.
    (You may do it in a `demo.ipynb` and submit it together with this
    package.)

### Part 4: Double Pendulum Lagrangian (1 point)

* Objective:
  Implement and solve the Lagrangian of a doublea pendulum problem.
* Details:

  * Define the Lagrangian of a two-dimensional double pendulum
    problem.
    Let's use the notations $theta1$ and $theta2$ in
    [wikipedia](https://en.wikipedia.org/wiki/Double_pendulum)
    and implement
    \begin{align}
      L = \frac{1}{6} ml^2 (\dot\theta_2^2 + 4\dot\theta_1^2 + 3\dot\theta_1\dot\theta_2\cos(\theta_1-\theta_2))
        + \frac{1}{2} mgl  (3\cos\theta_1 + \cos\theta_2)
    \end{align}
  * Use your `EL()` function (Part 3) to derive the equations of
    motion.
  * Integrate the system with your `RK4()` solver (Part 2).

### Part 5: Create a Movie (1 point)

* Objective:
  Combine everything into a Python script that generates an animation
  of the double pendulum.
* Details:
  * Write a script that:
    1. Accepts initial conditions for
       $(\theta_1,\theta_2,\dot\theta_1,\dot\theta_2)$.
    2. Evolves the double pendulum using your `RK4()` integrator.
    3. Produces a movie (e.g., using
       `matplotlib.animation.FuncAnimation()`) showing the double
       pendulum.
  * Install your script as an executable scirpt.
    See [this link](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#creating-executable-scripts).
  * Script prototype:
    ```bash
    double_pendulum x0=1 y0=0.5 vx0=0 vy0=0 t=20 dt=0.01 output="movie.mp4"
    ```


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
