# MBM_Kinematics_Model
python code for simulation a multi-backbone manipulator with two bendable segments
It implements the methodology presented in the following publication:

Zisos Mitros, Balint Thamo, Christos Bergeles, Lyndon Da Cruz, Kevin Dhaliwal, Mohsen Khadem, "Design and Modelling of a Continuum Robot for Distal Lung Sampling in Mechanically Ventilated Patients in Critical Care"


URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8126695/

If you enjoy this repository and use it, please cite our paper

@Article{Mitros2021,
  author                 = {Mitros, Zisos and Thamo, Balint and Bergeles, Christos and da Cruz, Lyndon and Dhaliwal, Kevin and Khadem, Mohsen},
  title                  = {Design and Modelling of a Continuum Robot for Distal Lung Sampling in Mechanically Ventilated Patients in Critical Care.},
  journal                = {Frontiers in robotics and AI},
  year                   = {2021},
  volume                 = {8},
  pages                  = {611866},
  article-doi            = {10.3389/frobt.2021.611866},
  title-abbreviation     = {Front Robot AI},
}

Dependencies: numpy, scipy, mpl_toolkits, matplotlib.

The module includes functions for modeling a multi-backbone manipulator with two bendable segments. It accepts joint variables q (pulling or pushing rods 1,2,4, and 5 plus insertino of stiff tube in the robot), length of first and second segment, input distributed force w and point force f, tolerance for solver.
Example.py shows how the module can be used to simulate the robot shape.
