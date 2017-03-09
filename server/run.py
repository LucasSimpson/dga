import json

from xor import XOR


def handler(params, lambda_):
    # get genotype from args
    json_str = params.get('json', None)
    json_data = json.loads(json_str)

    # build phenotype
    xor_solver = XOR(json_data['weights'])

    # evaluate
    fitness = xor_solver.evaluate()

    return fitness

if __name__ == '__main__':
    weights = {
        'weights': [
            1, 2, 3, 4,
            -1, -2, -3, -4,
            -4, -3, -2, -1
        ]
    }

    p = {
        'json': json.dumps(weights)
    }

    print(handler(p, None))
