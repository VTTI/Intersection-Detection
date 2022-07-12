#!/usr/bin/env python
import yaml
import subprocess

def get_updates(path):
    with open(path) as updates:
        configs = yaml.load(updates, Loader=yaml.FullLoader)
    optimizers =  configs['OPTIMIZERS']
    model_names = configs['MODELS']
    return optimizers, model_names

if __name__ == '__main__':
    optimizers, models = get_updates("./configs/config_updates.yaml")
    for mod in models:
        for optim in optimizers:
            print("Starting training {} with {} optimizer.".format(mod,optim))
            with open("./configs/config_baseline.yaml") as config_file:
                doc = yaml.load(config_file, Loader=yaml.FullLoader)
        
            doc['OPTIMIZER'] = optim
            doc['MODEL'] = mod

            with open("./configs/config_baseline.yaml","w") as config_file:
                yaml.dump(doc,config_file)
            subprocess.call(['python3','main.py'])
            
    

