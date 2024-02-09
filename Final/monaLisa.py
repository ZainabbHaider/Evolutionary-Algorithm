import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import jax.numpy as jnp
from jax import jit
from PIL import Image, ImageDraw
import concurrent.futures

POLY_MIN_POINTS = 3
POLY_MAX_POINTS = 7
POLYGONS = 50

class Chromosome(object):
    def __init__(self, imageSize, polygons=[]):
        self.imageSize = imageSize
        self.polygons = polygons

    def draw(self, show=False, save=False,generation=None):
        background = (0, 0, 0, 128) # black colour
        img = Image.new('RGB', self.imageSize, background)
        draw = Image.new('RGBA', self.imageSize)
        pdraw = ImageDraw.Draw(draw)
        for polygon in self.polygons:
            pdraw.polygon(polygon.points, fill=polygon.colour, outline=polygon.colour)
            img.paste(draw, mask=draw)

        if show:
            plt.imshow(img, cmap='gray')
            plt.axis('off')  # Turn off axis
            plt.show(block=False)
            plt.pause(1)
            plt.close()

        if save:
            directory_name = "Results"
            # Create the directory if it doesn't exist
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)
            fileName = u"{}".format(generation)
            saveLoc = os.path.join(directory_name, "{}.jpeg".format(fileName))
            img.save(saveLoc)

        return img

    def crossOver(self):
        # Create a deep copy of the polygons list
        polygons = [copy.deepcopy(polygon) for polygon in self.polygons]
        # Randomly select a polygon from the copied list
        rand_index = random.randrange(len(polygons))
        polygons[rand_index].mutate(self.imageSize)  # Mutate the selected polygon
        # Return a new Chromosome object with the mutated polygon
        return Chromosome(self.imageSize, polygons)
    
class Polygon(object):
    def __init__(self, colour=None, points=[]):
        self.colour = colour
        self.points = points

    def mutate(self, size):
        if random.random() <= 0.5:
            # Mutate color
            idx = random.randrange(4)
            value = random.randrange(256)
            self.colour = tuple(value if i == idx else c for i, c in enumerate(self.colour))
        else:
            # Mutate a random point
            idx = random.randrange(len(self.points))
            self.points[idx] = rand_point(size[0], size[1])

class Population(object):
    def __init__(self, populationSize, image) -> None:
        self.popSize = populationSize
        self.img = image
        
    def initialisePopulation(self):
        population = []
        pFit = []
        for i in range(self.popSize):
            chromosome = None
            polygons = []
            (width, height) = self.img.size

            for i in range(POLYGONS):
                pointNo = random.randrange(POLY_MIN_POINTS, POLY_MAX_POINTS + 1)
                points = []
                for j in range(pointNo):
                    point = rand_point(width, height)
                    points.append(point)
                
                colour = (255, 255, 255, 128) # white colour
                polygon = Polygon(colour, points)
                polygons.append(polygon)
            chromosome = Chromosome(self.img.size, polygons)
            population.append(chromosome)
        pFit = parallel_fitness(population, np.array(self.img))
        return population, pFit
     
    def populationCrossOver(self, parentChr, img):
        childChr = []
        cFit = []
        for i in range(self.popSize):
            childChromosome = parentChr[i].crossOver()
            childChr.append(childChromosome)
            # child = childChromosome.draw()
        cFit = parallel_fitness(childChr, np.array(img))
        return childChr, cFit

    def survivorSelection(self, parentChr, childChr, pFit, cFit):
        for i in range(self.popSize):    
            if cFit[i] < pFit[i]:
                parentChr[i] = childChr[i]
                pFit[i] = cFit[i]
        return parentChr, pFit, childChr, cFit
            
# Fitness function to calculate the difference between two images, using jax to speed up the process
@jit
def fitness(img1, img2):
    img1_array = jnp.array(img1) / 255.0
    img2_array = jnp.array(img2) / 255.0

    difference = jnp.abs(img1_array - img2_array)
    difference_sum = jnp.sum(difference, axis=(0, 1))
    fitness = jnp.sqrt(jnp.sum(difference_sum ** 2))
    return fitness

# computing fitness using parralel computing in order to speed up the process
def parallel_fitness(population, img_array):
    def evaluate_fitness(chromosome):
        image_array = np.array(chromosome.draw())
        return fitness(img_array, image_array)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        fitness_scores = list(executor.map(evaluate_fitness, population))
    return fitness_scores
            
# helper function to generate random points
def rand_point(width, height):
    x = random.randrange(0 - 10, width + 10, 1)
    y = random.randrange(0 - 10, height + 10, 1)
    return (x, y)

            
POPULATION_SIZE = 5
GENERATIONS = 100000
path = "mona.bmp"
img = Image.open(path)
# imageSize = img.size
population = Population(POPULATION_SIZE, img)
parentChr, pFit= population.initialisePopulation() # randomly initialise population
    

for generations in range(GENERATIONS+1):
    childChr, cFit = population.populationCrossOver(parentChr, img) # crossover
    parentChr, pFit, childChr, cFit = population.survivorSelection(parentChr, childChr, pFit, cFit) # survvior selection
    
    minFitness = pFit.index(min(pFit))
    print("Generation:", generations, "Best:", pFit[minFitness], "Average:", np.mean(pFit))
    if generations == GENERATIONS:
        print("Image at generation {}".format(generations))
        parentChr[minFitness].draw(show=True, save=False, generation=generations)
        print("Generation:", generations, "Best:", pFit[minFitness], "Average:", np.mean(pFit))
    elif generations% 2500 == 0:
        print("Image at generation {}".format(generations))
        parentChr[minFitness].draw(show=False, save=True, generation=generations)
        print("Generation:", generations, "Best:", pFit[minFitness], "Average:", np.mean(pFit))
      