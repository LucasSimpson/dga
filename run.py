from game.ga import GA2048


g = GA2048(pop_size=10)
m = g.generation()

print(m.fitness)

