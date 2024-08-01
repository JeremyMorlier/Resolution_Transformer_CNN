import os
import argparse

from simple_slurm import Slurm
from args import get_slurm_scheduler_argsparse
from references.common import create_dir, get_name

def extract_script_args(args, signal_id) :

    args_dict = args.__dict__
    args_names = list(args_dict.keys())

    # unwanted arguments (slurm arguments)
    excludes = ["add_resume", "job_name", "output", "error", "constraint", "nodes", "ntasks", "gres", "cpus_per_task", "time", "qos", "hint", "account", "script"]

    command_argument = ""

    for argument_name in args_names :
        if argument_name not in excludes :

            argument = args_dict[argument_name]
            if argument != None and argument != "":
                if type(argument) == bool :
                    if argument :
                        command_argument += "--" + argument_name + " "
                elif type(argument) == list :
                    list_string = ""
                    for element in argument :
                        list_string += " " + str(element)
                    command_argument += "--" + argument_name + list_string + " "
                else :
                    command_argument += "--" + argument_name + " " + str(argument) + " "
    command_argument += "--signal_id USR_%j "

    return command_argument

if __name__ == "__main__" :
    args, unknown_args = get_slurm_scheduler_argsparse().parse_known_args()

    signal_id = "USR_%j"

    # Slurm Sbatch setup
    slurm = Slurm(job_name=args.job_name,
                    output=args.output, error=args.error, 
                    constraint=args.constraint, nodes=args.nodes, ntasks=args.ntasks,
                    gres=args.gres, cpus_per_task=args.cpus_per_task, time=args.time, qos=args.qos, hint=args.hint, account=args.account, signal=signal_id +"@40")

    # usual commands
    slurm.add_cmd("module purge")
    slurm.add_cmd("conda deactivate")
    slurm.add_cmd("module load anaconda-py3/2023.09")
    slurm.add_cmd("conda activate $WORK/venvs/venvResolution")
    slurm.add_cmd("export WANDB_DIR=$WORK/wandb/")
    slurm.add_cmd("export WANDB_MODE=offline")

    script_args = extract_script_args(args, signal_id)
    
    # Add resume
    create_dir(args.output_dir)
    args.name = get_name(args)
    output_dir = os.path.join(args.output_dir, args.name)
    create_dir(output_dir)

    script_args = extract_script_args(args)

    slurm.sbatch(f'srun python3 {args.script}.py', script_args)
    
    # Save Slurm script
    script = slurm.script()
    if args.add_resume :
        script = script[:-1] + f" --resume {output_dir}/checkpoint.pth" + script[-1:]
    with open(output_dir + f"/slurm_script_job_{args.job_name}_.sh", "w") as file :
        file.writelines(script)
    print(script)