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

### Part 1: Setup Python Package (1 point)

* Objective:
  Turn your repo into a pip-installable python package with unit
  tests.
* Details:
  * Just like HW1 and HW2, create "pyproject.toml" and add a "LICENSE"
    file to this repository.
    This turns your homework set into a pip-installable python
    package, i.e., `pip install -e .` should work.
  * Add also a directory call `tests` and add a file
    `tests/test-p1.py` in it.
    Implement `test_import()` in `tests/test-p1.py` to make sure you
    can `import hw3`.
    You may look at how a
    [similar test](https://github.com/ua-2025q3-astr501-513/hw2/blob/main/.github/autograding/test-p1.py)
    was done in hw1 and 2.
  * Properly organize source code, tests, and documentations.

### Part 2: Implement RK4 Integrator (1 point)

* Objective:
  Implement a reusable 4th-order Runge–Kutta (RK4) integrator for systems of ODEs.
* Details:
  * Function prototype:
    ```python
    def RK4(
        f,   # the right hand side of the differential equation $dx/dt = f$
	x0,  # the initial dependent variable $x(t=t_0)$
	t0,  # the initial independent variable $t_0$
	dt,  # the (full) step size
	nt,  # how many steps you need to run
    ): ...
    ```
  * Demonstrate correctness by integrating a 1D harmonic oscillator
    and comparing against the analytic solution.
  * Add `tests/test-p1.py` to confirm behavior.

### Part 3: Euler-Lagrange via Autodiff (1 point)

* Objective:
  Use JAX automatic differentiation to obtain Euler-Lagrange equations
  from a given Lagrangian.
* Details:
  * Write a function:
    ```python
    def EL(L, q, qdot):
        ...
    ```
    where `L(q, qdot)` is a user-supplied Python function implementing
    the Lagrangian, and `q`, `qdot` are generalized coordinates and
    velocities.
  * Use `jax.grad` to compute the necessary derivatives for the
    Euler-Lagrange equation:
    \begin{align}
      \frac{d}{dt}\frac{\partial L}{\partial\dot{q}} - \frac{\partial L}{\partial q} = 0
    \end{align}
  * Verify on a single harmonic oscillator (compare with known
    analytic ODE).

### Part 4: Double Pendulum Lagrangian (1 point)

* Objective:
  Implement and solve the Euler-Lagrange equations for a
  two-dimensional double pendulum.
* Details:
  * Define the Lagrangian
  * Use your `EL()` function (Part 3) to derive the equations of
    motion.
  * Integrate the system with your `RK4()` solver (Part 2).
  * Add tests to check energy conservation and symmetry.

### Part 5: Create a Movie (1 point)

* Objective:
  Combine everything into a Python script that generates an animation
  of the system.
* Details:
  * Write a script that:
    1. Accepts initial conditions for $(x,y,\dot{x},\dot{y})$.
    2. Evolves the double pendulum using your RK4 integrator.
    3. Produces a movie (e.g., using
       matplotlib.animation.FuncAnimation) showing the trajectories.
  * Function prototype:
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
