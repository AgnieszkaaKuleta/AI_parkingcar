import numpy as np
import parking_model as pm
import neat
import os
import pickle
import math
STEP = 50
GENS = 50
POP_SIZE = 30
var_global = pm.GlobalVar()
class Car:
    def __init__(self, x, y, alpha, angle):
        self.x = x
        self.y = y
        self.alpha = alpha
        self.angle = angle
        self.rotation_center = []
        self.velocity = -2


def fitness(genomes, config):
    filename = 'data.txt'
    with open(filename, 'a') as file:
            nets = []
            cars = []
            ge = []
            indexes = []
            for genome_id, genome in genomes:
                indexes.append(genome_id)
                genome.fitness = 0
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                nets.append(net)
                cars.append(Car(10, 5, 0, 0))
                ge.append(genome)

            for x, car in enumerate(cars):
                for i in range(STEP):
                    state = [car.x, car.y, car.alpha]
                    angle = car.angle
                    if_collision = False
                    new_state = []
                    rotation_center = []

                    output = nets[cars.index(car)].activate(
                        (car.x, car.y, car.alpha, car.velocity))

                    decision = output.index(max(output))

                    new_state, rotation_center, if_collision = pm.model_of_car(state, car.angle, car.velocity , var_global)
                    if decision == 1:
                        car.angle -= np.pi/40

                    elif decision == 2:
                        car.angle += np.pi / 40

                    elif decision == 3:
                        car.velocity += 0.1

                    elif decision == 0:
                        car.velocity -= 0.1

                    car.x = new_state[0]
                    car.y = new_state[1]
                    car.alpha = new_state[2]
                    car.rotation_center = rotation_center
                    #print(x,i,state, car.angle , car.velocity)
                    if if_collision:
                        ge[cars.index(car)].fitness -= 50
                    ge[x].fitness = 100 - math.dist((car.x, car.y), (-1, -1)) - abs(car.alpha)

                    line = str(indexes[+x-POP_SIZE]) + " " + str(i) + " " + str (state[0]) + " " + str (state[1]) + " " + str (state[2]) + " " + str(car.angle) + " " + str(car.velocity) + "\n"
                    file.write(line)









def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('best.pickle')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(fitness, GENS)
    best = winner.key

    generation = math.ceil(best/POP_SIZE) + 1
    genome = best%POP_SIZE - 1
    print(best)
    print(generation, genome)
    print('\nBest genome:\n{!s}'.format(winner))
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

    lines_with_pattern = []
    with open('data.txt', 'r') as file:
        for line in file:
            if line.startswith(str(best)):
                lines_with_pattern.append(line)


    with open('history1.txt', 'w') as his:
        his.write(" ".join(str(elements) for elements in lines_with_pattern))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run_neat(config)



