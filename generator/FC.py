import numpy as np
import os
import sys
from collections import namedtuple

script_path = os.path.dirname(sys.argv[0])

def generate(filename, seed=1, n_customers=100, n_facilities=100, ratio=5):
    """
    Generate a Capacited Facility Location problem following
        Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.

bibtex
@article{cornuejols1991comparison,
  title={A comparison of heuristics and relaxations for the capacitated plant location problem},
  author={Cornu{\'e}jols, G{\'e}rard and Sridharan, Ranjani and Thizy, Jean-Michel},
  journal={European journal of operational research},
  volume={50},
  number={3},
  pages={280--297},
  year={1991},
  publisher={Elsevier}
}

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
    """

    rng = np.random.RandomState(seed)

    c_x = rng.rand(n_customers)
    c_y = rng.rand(n_customers)

    f_x = rng.rand(n_facilities)
    f_y = rng.rand(n_facilities)

    demands = rng.randint(5, 35+1, size=n_customers)
    capacities = rng.randint(10, 160+1, size=n_facilities)
    fixed_costs = rng.randint(100, 110+1, size=n_facilities) * np.sqrt(capacities) \
            + rng.randint(90+1, size=n_facilities)
    fixed_costs = fixed_costs.astype(int)

    total_demand = demands.sum()
    total_capacity = capacities.sum()

    # adjust capacities according to ratio
    capacities = capacities * ratio * total_demand / total_capacity
    capacities = capacities.astype(int)
    total_capacity = capacities.sum()

    # transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nobj:")
        file.write("".join([f" +{trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_facilities)]))
        file.write("".join([f" +{fixed_costs[j]} y_{j+1}" for j in range(n_facilities)]))

        file.write("\n\nsubject to\n")
        for i in range(n_customers):
            file.write(f"demand_{i+1}:" + "".join([f" -1 x_{i+1}_{j+1}" for j in range(n_facilities)]) + f" <= -1\n")
        for j in range(n_facilities):
            file.write(f"capacity_{j+1}:" + "".join([f" +{demands[i]} x_{i+1}_{j+1}" for i in range(n_customers)]) + f" -{capacities[j]} y_{j+1} <= 0\n")

        # optional constraints for LP relaxation tightening
        file.write("total_capacity:" + "".join([f" -{capacities[j]} y_{j+1}" for j in range(n_facilities)]) + f" <= -{total_demand}\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"affectation_{i+1}_{j+1}: +1 x_{i+1}_{j+1} -1 y_{j+1} <= 0\n")

        file.write("\nbounds\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"0 <= x_{i+1}_{j+1} <= 1\n")

        file.write("\nbinary\n")
        file.write("".join([f" y_{j+1}" for j in range(n_facilities)]))
        file.write("\nEnd")

def main():
    # 1. Create a folder called 'ca'
    if not os.path.exists('fc'):
        os.mkdir('fc')

    # # 2. Create a subfolder called 'train' and call the generate function 200 times
    train_folder = os.path.join('fc', 'train')
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)

    for i in range(1, 201):
        filename = os.path.join(train_folder, f'fc_{i}.lp')
        generate(filename, seed=i+135, n_customers=200, n_facilities=100, ratio=5)

    # 3. Create a subfolder called 'test' and call the generate function 100 times
    test_folder = os.path.join('fc', 'test')
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)

    for i in range(1, 101):
        filename = os.path.join(test_folder, f'fc_{i}.lp')
        generate(filename, seed=i+175, n_customers=200, n_facilities=100, ratio=5)

if __name__ == '__main__':
    main()
# if __name__ == "__main__":
#     FacilityLocationDistribution = namedtuple("FacilityLocationDesc", [
#         "distribution_name",
#         "num_of_instances",
#         "n_facilities",
#         "n_customers",
#         "seed"
#     ])

#     distributions = [
#         FacilityLocationDistribution(
#             distribution_name="train_easy",
#             num_of_instances=200,
#             n_facilities=100,
#             n_customers=200,
#             seed=564
#         ),

#         FacilityLocationDistribution(
#             distribution_name="train_hard",
#             num_of_instances=200,
#             n_facilities=100,
#             n_customers=400,
#             seed=763
#         ),

#         FacilityLocationDistribution(
#             distribution_name="test_easy",
#             num_of_instances=100,
#             n_facilities=100,
#             n_customers=200,
#             seed=895
#         ),

#         FacilityLocationDistribution(
#             distribution_name="test_hard",
#             num_of_instances=100,
#             n_facilities=100,
#             n_customers=400,
#             seed=459
#         )
#     ]

#     instance_dir = os.path.join(script_path, "instances")
#     if not os.path.exists(instance_dir):
#         os.makedirs(instance_dir)

#     for inst_desc in distributions:

#         dist_desc = "facility_location_{}_nf_{}_nc_{}_seed_{}".format(
#             inst_desc.distribution_name, inst_desc.n_facilities, inst_desc.n_customers,
#             inst_desc.seed
#         )
#         distribution_dir = os.path.join(instance_dir,
#                                         dist_desc)
#         # generate random instances
#         for i in range(inst_desc.num_of_instances):
#             outfile = os.path.join(distribution_dir, dist_desc+"_{}.lp".format(i))
#             os.makedirs(os.path.dirname(outfile), exist_ok=True)
#             generate(outfile, inst_desc.seed, inst_desc.n_customers, inst_desc.n_facilities, 5)