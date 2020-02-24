import argparse
from contexts import core

from core.FN.policy_gradient_agent import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PG Agent")
    parser.add_argument("--play",
                        action="store_true",
                        help="play with trained model")

    args = parser.parse_args()
    main(args.play)
