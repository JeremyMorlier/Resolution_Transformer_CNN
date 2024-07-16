import argparse

from simple_slurm import Slurm
from args import get_slurm_scheduler_argsparse

def extract_script_args(args) :

    args_dict = args.__dict__
    args_names = list(args_dict.keys())

    # unwanted arguments (slurm arguments)
    excludes = ["job_name", "output", "error", "constraint", "nodes", "ntasks", "gres", "cpus_per_task", "time", "qos", "hint", "account", "script"]

    command_argument = ""

    for argument_name in args_names :
        if argument_name not in excludes :

            argument = args_dict[argument_name]
            if argument != None and argument != "":
                if type(argument) == bool :
                    if argument :
                        command_argument += "--" + argument_name + " "
                else :
                    command_argument += "--" + argument_name + " " + str(argument) + " "

    return command_argument

if __name__ == "__main__" :
    args = get_slurm_scheduler_argsparse().parse_args()

    # Slurm Sbatch setup
    slurm = Slurm(job_name=args.job_name,
                    output=args.output, error=args.error, 
                    constraint=args.constraint, nodes=args.nodes, ntasks=args.ntasks,
                    gres=args.gres, cpus_per_task=args.cpus_per_task, time=args.time, qos=args.qos, hint=args.hint, account=args.account)

    # usual commands
    slurm.add_cmd("module purge")
    slurm.add_cmd("conda deactivate")
    slurm.add_cmd("module load anaconda-py3/2023.09")
    slurm.add_cmd("conda activate $WORK/venvs/venvResolution")
    slurm.add_cmd("export WANDB_DIR=$WORK/wandb/")
    slurm.add_cmd("export WANDB_MODE=offline")

    script_args = extract_script_args(args)

    slurm.sbatch(f'srun python3 {args.script}.py', script_args)
    print(slurm)