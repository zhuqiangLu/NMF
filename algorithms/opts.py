import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root',
        type=str,
        default="../data/CroppedYaleB")
    
    parser.add_argument(
        '--reduce',
        type=int,
        default=2
    )
    parser.add_argument(
        '--split_ratio',
        type=float,
        default=0.9
    )
    

    args = parser.parse_args()
    return args