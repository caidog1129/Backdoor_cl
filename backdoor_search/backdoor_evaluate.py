import pandas as pd
import gurobipy as grb
from ast import literal_eval

import argparse
import os
import glob

def main(instance_dir):
    env = grb.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.setParam("Threads",1)
    env.start()
    
    if glob.glob('%s/*.mps*' % instance_dir):
        instance_path = glob.glob('%s/*.mps*' % instance_dir)[0]
    else:
        instance_path = glob.glob('%s/*.lp*' % instance_dir)[0]
    m = grb.read(instance_path, env=env)
    m.optimize()
    baseline = m.Runtime
    print(m.Runtime)

    df = pd.read_csv(glob.glob('%s/*.csv*' % instance_dir)[0], sep=";")
    for index, row in df.sort_values("reward", ascending = False).iloc[:50].iterrows():
        m = grb.read(instance_path, env=env)
        for i in range(len(m.getVars())):
            if i in literal_eval(row["backdoor_list"]):
                m.getVars()[i].BranchPriority = 2
            else:
                m.getVars()[i].BranchPriority = 1
        m.update()
        m.optimize()
        df.loc[index,"run_time"] = m.Runtime
        print(m.Runtime)

    # for index, row in df[df["reward"] == 0].sample(50).iterrows():
    #     for i in range(len(m.getVars())):
    #         if i in literal_eval(row["backdoor_list"]):
    #             m.getVars()[i].BranchPriority = 2
    #         else:
    #             m.getVars()[i].BranchPriority = 1
    #     m.update()
    #     m.optimize()
    #     df.loc[index,"run_time"] = m.Runtime
    #     print(m.Runtime)
            
    df = df.dropna()
    df = df.sort_values("run_time")
    df.to_csv(os.path.join(instance_dir, "backdoor_evaluate_sampling" + str(baseline) + ".csv"))
    
if __name__ == '__main__':
    parser_main = argparse.ArgumentParser()

    parser_main.add_argument("--instance_dir", type=str)
    args_main = parser_main.parse_args()
    print(args_main)

    main(instance_dir=args_main.instance_dir)

