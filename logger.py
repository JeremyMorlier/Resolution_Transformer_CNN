import wandb
import json
import os

from references.common import create_dir

class Logger() :


    def __init__(self, project_name, run_name, tags = None, resume = False, id=None, args = None, mode="txt", log_dir="./") :

        self.step = 0
        self.mode = mode
        self.id = id

        if mode == "wandb" :
            
            wandb_resume = None
            if resume :
                wandb_resume = "allow"
            wandb.init(
                # set the wandb project where this run will be logged
                project=project_name,
                name=run_name,
                tags=tags,
                resume=wandb_resume,
                id=id,
                # track hyperparameters and run metadata
                config=args
            )
        elif self.mode == "txt" :

            create_dir(log_dir)
            self.file = os.path.join(log_dir, project_name + "_" + run_name + ".log")

            with open(self.file, "a") as file :
                header = {"project_name": project_name, "run_name": run_name, "tags": tags, "args": args.__dict__}

                json.dump(header, file)
                file.write("\n")
    def log(self, dictionnary) :
        # Update log with global step to better sync between modes
        dictionnary["step"] = self.step
        self.step += 1

        if self.mode == "wandb" :
            wandb.log(dictionnary)
        elif self.mode == "txt" :
            with open(self.file, "a") as file :
                json.dump(dictionnary, file)
                file.write("\n")
    def finish(self) :
        if self.mode == "wandb" :
            wandb.finish()
        
        print("Logging Finished !")
