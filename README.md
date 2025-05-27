# MSMFE_MFMFE_Methods
This repository contains the source code for the numerical experiments conducted as part of my PhD thesis at the University of Pittsburgh. It implements MSMFE (Multipoint Stress Mixed Finite Element) methods for linear elasticity, and MSMFE-MFMFE (Multipoint Stress–Multipoint Flux Mixed Finite Element) methods for the Biot system of poroelasticity, on cuboid and distorted quadrilateral grids.

## Authors

- **Ilona Ambartsumyan**, **Eldar Khattatov** (2016–2020)  
- **Ibrahim Yazici** (2023–2025)

## Installing deal.II

This code is written in C++ and uses version `9.5.1` of the [deal.II finite element library](https://www.dealii.org/).

🔗 Official download page: [https://www.dealii.org/download.html](https://www.dealii.org/download.html)

## Notes on Running the Experiments

The `Code` folder contains the main source files needed to run the simulations using deal.II.

### Running the Program

Once the project is built, you can run the executable with:

```bash
make run
````

You will first be prompted to select the model:

```
Choose the model to run:
  1: Mixed Darcy Problem
  2: Multipoint Mixed Darcy Problem
  3: Mixed Elasticity Problem
  4: Multipoint Mixed Elasticity Problem
  5: Mixed Biot Problem
```

After selecting a model, you will be asked to specify the dimension (`2d` or `3d`).

### Parameter Files

The parameter files (`.prm`) located in the `Code` directory define the simulation setup, including:

* Grid types
* Mesh refinements
* Boundary and initial conditions
* Physical parameter fields
* Exact solutions (for verification)

You can modify these files to run different problems or to reproduce the test cases presented in **Chapters 3–7** of the thesis.

### Chapter-Specific Instructions

* **Chapter 3, Example 3:**
  Use the `.prm` files from the `Chapter3_Example3_parameters` folder.
  Replace the existing `parameters_elasticity.prm` file in the `Code` directory with the relevant file from that folder.

* **Chapter 3, Example 2 (MSMFE-0 method):**
  Use the `.prm` and source files from the `Chapter3_Example2_MSMFE-0` folder.
  Replace both:

  * `parameters_elasticity.prm`
  * `elasticity_mfe.cpp`
    in the `Code` directory with the versions provided in that folder.

---

These changes allow you to reproduce the results shown in the thesis by simply rerunning the executable with the appropriate configuration.








