import json

from xor import Game2048


def handler(params, lambda_):
    # get genotype from args
    json_str = params.get('json', None)
    json_data = json.loads(json_str)

    # build phenotype
    xor_solver = Game2048(json_data['weights'])

    # evaluate
    score = 0
    for i in range(10):
        score += xor_solver.evaluate()

    return score / 10

if __name__ == '__main__':
    import random, math

    for a in range (50):
        w = {'weights': [random.normalvariate(0, 1) for i in range(Game2048.size)]}

        p = {
            'json': json.dumps(w)
        }

        print(math.sqrt(handler(p, None)))
