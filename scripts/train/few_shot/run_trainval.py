import argparse

from trainval import main

parser = argparse.ArgumentParser(description='Re-run prototypical networks training in trainval mode')

model_path = r'F:\jupyter_notebook\prototypical-networks\scripts\train\few_shot\results\20231015-1934best_model.pt'
parser.add_argument('--model.model_path', type=str, default=model_path, metavar='MODELPATH',
                    help="location of pretrained model to retrain in trainval mode")

args = vars(parser.parse_args())

main(args)
