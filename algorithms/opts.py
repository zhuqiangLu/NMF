import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    '''
    ==========================
    |         Training        |
    ==========================
    '''
    parser.add_argument(
        '--epoch',
        type=int,
        default=5
    )

    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=10
    )

    parser.add_argument(
        '--iters',
        type=int,
        default=1
    )

    parser.add_argument(
        '--tol',
        type=int,
        default=1e-7
    )

    



    '''
    ==========================
    |         NMF            |
    ==========================
    '''
    parser.add_argument(
        '--NMF_OBJ',
        type=str,
        default="ROBUSTL1"
    )




    '''
    ==========================
    |         DATA           |
    ==========================
    '''

    parser.add_argument(
        '--root',
        type=str,
        default="../data/CroppedYaleB")
    
    parser.add_argument(
        '--reduce',
        type=int,
        default=4
    )
    parser.add_argument(
        '--split_ratio',
        type=float,
        default=0.9
    )

    '''
    ==========================
    |         NOISE          |
    ==========================
    '''

    parser.add_argument(
        '--noise',
        type=str,
        default='gaussian'
    )


    # salt and pepper parameters
    # ratio of noise
    parser.add_argument(
        '--p',
        type=float,
        default=0.2
    )
    # ratio of salt
    parser.add_argument(
        '--r',
        type=float,
        default=0.2
    )

    # guassian white noise
    # mean
    parser.add_argument(
        '--mu',
        type=int,
        default=0
    )
    # std div
    parser.add_argument(
        '--sigma',
        type=int,
        default=1
    )
    # k
    parser.add_argument(
        '--k',
        type=int,
        default=64
    )
    
    

    args = parser.parse_args()
    return args