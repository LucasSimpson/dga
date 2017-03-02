from game.ga import GA2048


g = GA2048(pop_size=50)

while True:
    m = g.generation()
    print(m.fitness)

