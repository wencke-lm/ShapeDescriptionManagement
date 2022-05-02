"""create_report.py - provides a commandline interface to metrics"""

import argparse
import logging
import os
import sys

import scipy as sp

# in order to access module from sister directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from lib.dataloader import write_complexity_report


if __name__ == "__main__":
    # create commandline parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Compute a selection of complexity measures.\n"
            "+ Mean Self-BLEU\n"
            "+ Mean Subgraph Density\n"
            "+ Minimum Hellinger Distance\n"
            "+ Geometric Separability Index\n"
            "+ Imbalance Ratio\n"
            "+ Subset Representativity\n"
            "+ Number of Classes\n"
            "+ Number of Instances\n"
        ),
        epilog="All files are expected to be utf-8 encoded."
    )

    # add syntax definition
    parser.add_argument(
        "--source", nargs=1, metavar="FILE",
        help=(
            "path to a directory that includes\n"
            "several data sets as nested directories"
        )
    )
    parser.add_argument(
        "--output", nargs=1, metavar="DIR", default= ".",
        help=(
            "directory that a tsv-file, called\n"
            "'complexity_report.tsv' will be saved to,\n"
            "if a file of the same name already exists, it \n"
            "will be overriden, defaults to current directory"
        )
    )

    # configure logger
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    # parse commandline input
    if len(sys.argv) == 1:
        parser.print_help()
    else:
        args = parser.parse_args()
        write_complexity_report(*args.source, *args.output)
