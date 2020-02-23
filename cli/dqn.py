import argparse
from contexts import core
from core.FN.dqn_agent import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DQN Agent")
    parser.add_argument("--play",
                        action="store_true",
                        help="play with trained model")
    parser.add_argument("--test",
                        action="store_true",
                        help="train by test mode")

    args = parser.parse_args()
    main(args.play, args.test)
