import copy
import os.path
import random
from operator import itemgetter
import faulthandler
import retro
import pyaudio
import numpy as np
from timeit import default_timer
import tensorflow as tf
import pickle
import PySimpleGUI as sg
from PIL import Image, ImageTk



class sound:
    def play(self, array, audio):
        audio.write(array)



def main():
    faulthandler.enable()
    save_file = 'sonicAI.pkl'

    obs = env.reset()
    observer = gameObserver()

    num_actions = env.action_space.n
    state_size = np.shape(obs)
    print(state_size)
    tf.compat.v1.disable_eager_execution()
    currentGen = 0
    maxGen = 0
    genController = generationController(baseScore=480)
    if os.path.exists(save_file):
        with open(save_file, 'rb') as inp:
            genController = pickle.load(inp)
    nn = neuralNetwork(currentGen, state_size, num_actions)

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=2,
                    rate=44100,
                    output=True,
                    stream_callback=callback)
    stream.start_stream()

    game_view_column = [[sg.Image(key="-IMAGE-", size=(640, 480)),]]
    fitness_list_column = [[sg.Text(key="-FITNESS-LIST-"),]]
    fitness_details_column = [[sg.Text(key="-FITNESS-")], [sg.Text(key="-MAX-FITNESS-")], [sg.Text(key="-GENS-")], [sg.Text(key="-PARENTS-")]
                              ]

    layout = [
        [sg.Column(game_view_column),
        sg.Column(fitness_list_column),
        sg.Column(fitness_details_column)],
        [sg.Image(key="d-pad"),sg.Image(key="buttons")],
    ]

    window = sg.Window("Sonic AI", layout, size=(600, 350))
    event, values = window.read(timeout=1)
    d_pad_image = Image.open("d-pad.png")
    d_pad_up = Image.open("up.png")
    d_pad_down = Image.open("down.png")
    d_pad_left = Image.open("left.png")
    d_pad_right = Image.open("right.png")

    button_image = Image.open("buttons.png")
    button_a = Image.open("a-button.png")
    button_b = Image.open("b-button.png")
    button_c = Image.open("c-button.png")
    button_start = Image.open("start-button.png")

    last_lives = 3

    while True:
        start = default_timer()
        action = nn.predict(obs)

        obs, rew, done, info = env.step(action[0])

        nn.setScore(observer.calcReward(info['x'], info['rings'], info['score']))
        end_of_level = info['screen_x_end']
        level_complete =False
        if info['x'] >= end_of_level:
            done = True
            level_complete = True

        d_pad_image_buffer = d_pad_image.copy()
        buttons_image_buffer = button_image.copy()

        #actions = action_to_array(action[0])
        if action[0][4] >= 1.0:
            d_pad_image_buffer.paste(d_pad_up, (0, 0), mask=d_pad_up)

        if action[0][5] >= 1.0:
            d_pad_image_buffer.paste(d_pad_down, (0, 0), mask=d_pad_down)

        if action[0][6] >= 1.0:
            d_pad_image_buffer.paste(d_pad_left, (0, 0), mask=d_pad_left)

        if action[0][7] >= 1.0:
            d_pad_image_buffer.paste(d_pad_right, (0, 0), mask=d_pad_right)

        window["d-pad"].update(data=ImageTk.PhotoImage(d_pad_image_buffer))

        if action[0][0] >= 1.0:
            buttons_image_buffer.paste(button_b, (0,0), mask=button_b)

        if action[0][1] >= 1.0:
            buttons_image_buffer.paste(button_a, (0,0), mask=button_a)

        if action[0][8] >= 1.0:
            buttons_image_buffer.paste(button_c, (0,0), mask=button_c)

        window["buttons"].update(data=ImageTk.PhotoImage(buttons_image_buffer))


        event, values = window.read(timeout=1)
        new_image = Image.fromarray(env.render(mode="rgb_array"), 'RGB')

        image = ImageTk.PhotoImage(image=new_image)

        window["-IMAGE-"].update(data=image)
        window["-FITNESS-"].update("Fitness: " + str(observer.calcReward(info['x'], info['rings'], info['score'])))


        if done or observer.stagnant() or info['lives'] < last_lives:
            genController.loadAncestor(nn.score, nn.get_weights(), level_complete)

            window["-GENS-"].update("Generation: " + str(genController.getGen()))
            window["-MAX-FITNESS-"].update("Max Fitness: " + str(genController.getMaxFitness()))
            obs = env.reset()
            currentGen += 1
            maxGen += 1
            observer = gameObserver()
            if genController.isMaxAncestor():
                observer.increaseStagnation()
            tf.keras.backend.clear_session()
            nn = neuralNetwork(currentGen, state_size, num_actions)
            nn.setAncestors(genController.getAncestor(window))
            window["-FITNESS-LIST-"].update(genController.getFitnessList())
            if currentGen % 10 == 0:
                with open(save_file, 'wb') as outp:
                    pickle.dump(genController, outp, pickle.HIGHEST_PROTOCOL)
        last_lives = info['lives']



    env.close()


class generationController():
    def __init__(self, baseScore = 0, gensKept = 10, maxMutation = 0.01, currentGen = 0):
        self.baseScore = baseScore
        self.ancestors = []
        self.generationsToKeep = gensKept
        self.maxMutation = maxMutation
        self.currentGen = currentGen

    def getMaxFitness(self):
        if len(self.ancestors) > 0:
            return self.ancestors[0][0]
        return self.baseScore

    def isMaxAncestor(self):
        return len(self.ancestors) == self.generationsToKeep

    def getAncestor(self, window):
        if len(self.ancestors) < self.generationsToKeep:
            print("not enough ancestors yet!")
            print(len(self.ancestors))
            return []

        mixAncestors = random.randint(1, 10)

        #ancestor1 = random.randint(0, len(self.ancestors)-1)
        ancestor1 = random.randint(0, len(self.ancestors)-1)

        ancestor2 = random.randint(0, len(self.ancestors)-1)
        while ancestor2 == ancestor1:
            ancestor2 = random.randint(0, len(self.ancestors) - 1)


        returnAncestor = copy.deepcopy(self.ancestors[ancestor1])
        ancestorGen = returnAncestor[2]
        mixingAcnestor = self.ancestors[ancestor2]
        winText = "Parents:\n" + str(returnAncestor[0]) +"\n"
        print("basing ancestor on fitness: " + str(returnAncestor[0]))
        if mixAncestors < 6:
            print("ancestor mixing...")
            print("mixing ancestor: " + str(mixingAcnestor[0]))
            winText = winText + str(mixingAcnestor[0])
        outerCount = 0
        for outer in returnAncestor[1]:
            le = len(outer)
            if len(outer) > 0 and len(outer[0]) < 33:
                innerCount = 0
                for inner in outer[0]:
                    le = len(inner)
                    index = 0
                    for weight in inner:
                        mixedWeight = weight
                        if mixAncestors < 6 and random.randint(0, 1) == 1:
                            mixedWeight = mixingAcnestor[1][outerCount][0][innerCount][index]
                        mutation = random.uniform(-1*self.maxMutation * (self.currentGen - ancestorGen), self.maxMutation * (self.currentGen - ancestorGen))

                        inner[index] = mixedWeight + mutation
                        index+=1
                innerCount += 1
            outerCount += 1

        window["-PARENTS-"].update(winText)
        return returnAncestor

    def loadAncestor(self, score, weights, level_complete):
        self.currentGen += 1
        if score > self.baseScore:
            for ancestor in self.ancestors:
                if ancestor[0] == score:
                    print("skipping same ancestor")
                    return
            self.ancestors.append([score, weights, self.currentGen-1, level_complete])
            self.ancestors = sorted(self.ancestors, key=itemgetter(0), reverse=True)
            if len(self.ancestors) > self.generationsToKeep :
                self.ancestors = self.ancestors[:self.generationsToKeep]

    def getGen(self):
        return self.currentGen

    def getFitnessList(self):
        returnValue = "Top Fitness:\n"
        for entry in self.ancestors:
            returnValue = returnValue + str(entry[0]) + ("★\n" if entry[3] else "\n")
        return returnValue


class gameObserver():
    def __init__(self, rings = 0):
        self.previousRings = rings
        self.previousScore = 0
        self.stagnant_count = 0
        self.stagnant_max = 60
        self.stagnant_after_maxAncestor = 1800

    def calcReward(self, sonicX, rings, gamescore):
        ringsAmt = 0

        if rings > self.previousRings:
            ringsAmt = rings - self.previousRings

        score = (sonicX * 5) + (ringsAmt * 10) + (gamescore)

        if(score == self.previousScore):
            self.stagnant_count += 1
        else:
            self.stagnant_count = 0

        self.previousScore = score
        return score

    def increaseStagnation(self):
        self.stagnant_max = self.stagnant_after_maxAncestor

    def stagnant(self):
        return self.stagnant_count > self.stagnant_max


class neuralNetwork():
    def __init__(self, gen, state_size, num_actions):
        self.generation = gen
        self.score = 0
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=state_size),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_actions, activation='sigmoid')
            ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def show_model(self):
        return tf.keras.utils.model_to_dot(self.model, show_shapes=True)

    def get_weights(self):
        weights = []
        for w in self.model.layers:
            weights.append((w.get_weights()))

        return weights

    def predict(self, obs):
        return self.model.predict(np.array(obs).reshape([1, 65536]))
        #return self.model.predict(np.array(obs).reshape([1, 224, 320, 3]))


    def setScore(self, score):
        self.score = score

    def mutate(self):
        mutation = random.uniform(-2, 2)
        self.model.layers.multiply(mutation)

    def setAncestors(self, nnParents):
        if(len(nnParents) == 0):
            return


        curr = 0
        for layer in self.model.layers:
            weights = nnParents[1][curr]
            layer.set_weights(weights)
            curr += 1


def action_to_array(a):
    actions = []
    for p in range(1):
        action = 0
        ap = a[12 * p:12 * (p + 1)]
        for i in range(len(ap)):
            action |= int(ap[i]) << i

        ap = np.zeros([12], np.uint8)
        for i in range(12):
            ap[i] = (action >> i) & 1
        actions.append(ap)
    return actions


def callback(in_data, frame_count, time_info, status):
    a = env.em.get_audio()
    if(len(a)):
        return (a, pyaudio.paContinue)
    return (bytes(), pyaudio.paContinue)

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    env = retro.make(game='SonicTheHedgehog2-Genesis', obs_type=retro.Observations.RAM)
    main()
