import argparse
from core.FN.value_function_agent import main


def test_FN_value_function_agent():
    parser = argparse.ArgumentParser(description="VF Agent")
    parser.add_argument("--play",
                        action="store_true",
                        help="play with trained model")
    args = parser.parse_args()
    main(args.play)
