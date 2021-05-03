"""
Module for argument parcer.
Many of the arguments are from Huggingface's run_squad example:
https://github.com/huggingface/transformers/blob/7972a4019f4bc9f85fd358f42249b90f9cd27c68/examples/run_squad.py
"""
import argparse
import os

args = argparse.ArgumentParser(description='indirect qa')



args.add_argument('--experiment',
                  type=str,
                  default='testing',
                  help='name of experiment')
args.add_argument('--save_dir',
                  type=str,
                  default='results',
                  help='directory to save results')
# args.add_argument('--plots_dir',
#                    type=str,
#                    default='plot',
#                    help='directory to save results')
args.add_argument('--seed',
                  type=int,
                  default=42,
                  help='random seed')
args.add_argument('--run_log',
                  type=str,
                  default=os.path.join(os.getcwd(),'log'),
                  help='where to print run log')
args.add_argument('--access_mode',
                  type=int,
                  default=0o777,
                  help='access mode of files created')
# =============================================================================
# for dataloading
# =============================================================================

args.add_argument('--data_dir',
					type=str,
					default='data',
					help='directory storing all data')

# =============================================================================
# for training
# =============================================================================
args.add_argument('--logging_steps',
					type=int,
					default=1e4,
					help='logs best weights every X update steps for experiment')


def check_args(parser):
    """
    make sure directories exist
    """
    assert os.path.exists(parser.data_dir), "Data directory does not exist"
    assert os.path.exists(parser.save_dir), "Save directory does not exist"
    assert os.path.exists(parser.run_log),  "Run logging directory does not exist"