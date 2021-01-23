#!/usr/bin/env python

import os, io, argparse, numpy as np
from sklearn.externals.joblib import Parallel, delayed
from PIL import Image
import matplotlib.pyplot as plt
import gym

# Disable TensorFlow GPU for parallel execution.
if os.name == 'nt':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
from tensorflow.python import keras as K


class EvolutionalAgent():

    def __init__(self, actions):
        self.actions = actions
        self.model = None

    def save(self, model_path):
        self.model.save(model_path, overwrite=True, include_optimizer=False)

    @classmethod
    def load(cls, env, model_path):
        actions = list(range(env.action_space.n))
        agent = cls(actions)
        agent.model = K.models.load_model(model_path)
        return agent

    def initialize(self, state, weights=()):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(K.layers.Conv2D(
            3, kernel_size=5, strides=3,
            input_shape=state.shape, kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(len(self.actions), activation="softmax"))
        self.model = model
        if len(weights) > 0:
            self.model.set_weights(weights)
    
    def policy(self, state):
        action_probs = self.model.predict(np.array([state]))[0]
        action = np.random.choice(self.actions, size=1, p=action_probs)[0]

        return action
    
    def play(self, env, episode_count=5, render=True):

        for e in range(episode_count):
            s = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if render:
                    env.render()
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)
                episode_reward += reward
                s = n_state
            else:
                print(f"Get reward {episode_reward}")


class Observer():

    def __init__(self, env):
        self._env = env

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        return self.transform(self._env.reset())

    def render(self):
        self._env.render(mode="human")

    def step(self, action):
        n_state, reward, done, info = self._env.step(action)
        return self.transform(n_state), reward, done, info

    def transform(self, state):
        raise NotImplementedError("You have to implement transform method.")
        
class CatcherObserver(Observer):

    def __init__(self, width, height, frame_count):
        import gym_ple
        self._env = gym.make("Catcher-v0")
        self.frame_count = frame_count
        self.width = width
        self.height = height

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        return self.transform(self._env.reset())

    def render(self):
        self._env.render(mode="human")

    def step(self, action):
        n_state, reward, done, info = self._env.step(action)
        return self.transform(n_state), reward, done, info

    def transform(self, state):
        grayed = Image.fromarray(state).convert("L")
        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0  # scale to 0-1.
        normalized = np.expand_dims(normalized, axis=2)  #  H x W => W x W x C
        return normalized
    
# class Trainer():
#     def __init__(self, buffer_size=1024, batch_size=32, gamma=0.9, report_interval=10, log_dir=""):
#         self.buffer_size = buffer_size
#         self.batch_size = batch_size
#         self.gamma = gamma
#         self.report_interval = report_interval
#         self.logger = Logger(log_dir, self.trainer_name)
#         self.experiences = deque(maxlen=buffer_size)
#         self.training = False
#         self.training_count = 0
#         self.reward_log = []

#     @property
#     def trainer_name(self):
#         class_name = self.__class__.name
#         snaked = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
#         snaked = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snaked).lower()
#         snaked = snaked.replace("_trainer", "")
#         return snaked
    
#     def train_loop(self, env ,agent, episode=200, initial_count=-1,
#     render=False, observe_interval=0):
#         self.experiences = deque(maxlen=self.buffer_size)
#         self.training = False
#         self.training_count = 0
#         self.reward_log = []
#         frames = []
    
#         for i in range(episode):
#             s = env.reset()
#             done = False
#             step_count = 0
#             self.episode_begin(i, agent)
#             while not done:
#                 if render:
#                     env.render()
#                 if self.training and observe_interval > 0 and \
#                     (self.training_count == 1 or self.training_count % observe_interval == 0):
#                     frames.append(s)
#                 a = agent.policy(s)
#                 n_state, reward, done, info = env.step(a)
#                 e = Experience(s, a, reward, n_state, done)
#                 self.experiences.append(e)
#                 if not self.training and \
#                     len(self.experiences) == self.buffer_size:
#                     self.begin_train(i, agent)
#                     self.training = True
#                 s = n_state
#                 step_count += 1
#             else:
#                 self.episode_end(1, step_count, agent)

#                 if not self.training and \
#                     initial_count > 0 and i >= initial_count:
#                     self.begin_train(i, agent)
#                     self.training = True
                
#                 if self.training:
#                     if len(frames) > 0:
#                         self.logger.write_image(self.training_count, frames)
#                         frames = []
#                     self.training_count += 1
    
#     def episode_begin(self, episode, agent):
#         pass

#     def begin_train(self, episode, agent):
#         pass

#     def step(self, episode, step_count, agent, experience):
#         pass

#     def episode_end(self, episode, step_count, agent):
#         pass

#     def is_event(self, count, interval):
#         return True if count != 0 and count % interval == 0 else False

#     def get_recent(self, count):
#         recent = range(len(self.experiences) - count, len(self.experiences))
#         return [self.experiences[i] for i in recent]

class EvolutionalTrainer():

    def __init__(self, population_size=20, sigma=0.5, learning_rate=0.1,
                 report_interval=10, log_dir=""):
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.weights = ()
        self.reward_log = []
        self.report_interval=report_interval
#        self.logger = Logger(log_dir, self.trainer_name)

    def train(self, epoch=100, episode_per_agent=1, render=False):
        env = self.make_env()
        actions = list(range(env.action_space.n))
        s = env.reset()
        agent = EvolutionalAgent(actions)
        agent.initialize(s)
        self.weights = agent.model.get_weights()

        with Parallel(n_jobs=-1) as parallel:
            for e in range(epoch):
                experiment = delayed(EvolutionalTrainer.run_agent)
                results = parallel(experiment(
                    episode_per_agent, self.weights, self.sigma)
                    for p in range(self.population_size))
                self.update(results)
                self.log()
        agent.model.set_weights(self.weights)
        return agent
    
    @classmethod
    def make_env(cls):
        return CatcherObserver(width=50, height=50, frame_count=5)

    @classmethod
    def run_agent(cls, episode_per_agent, base_weights, sigma, max_step=1000):
        env = cls.make_env()
        actions = list(range(env.action_space.n))
        agent = EvolutionalAgent(actions)

        noises = []
        new_weights = []

        # Make weight.
        for w in base_weights:
           noise = np.random.randn(*w.shape)
           new_weights.append(w + sigma * noise)
           noises.append(noise)
        
        # Test Play.
        total_reward = 0
        for e in range(episode_per_agent):
            s = env.reset()
            if agent.model is None:
                agent.initialize(s, new_weights)
            done = False
            step = 0
            while not done and step < max_step:
                a = agent.policy(s)
                n_state, reward, done, info = env.step(a)
                total_reward += reward
                s = n_state
                step += 1
        
        reward = total_reward / episode_per_agent
        return reward, noises

    def update(self, agent_results):
        rewards = np.array([r[0] for r in agent_results])
        noises = np.array([r[1] for r in agent_results])
        normalized_rs = (rewards - rewards.mean()) / rewards.std()

        # Update base weights.
        new_weights = []
        for i, w in enumerate(self.weights):
            noise_at_i = np.array([n[i] for n in noises])
            rate = self.learning_rate / (self.population_size * self.sigma)
            w = w + rate * np.dot(noise_at_i.T, normalized_rs).T
            new_weights.append(w)
        
        self.weights = new_weights
        self.reward_log.append(rewards)
    
    def log(self):
        rewards = self.reward_log[-1]
        print(f"Epoch {len(self.reward_log)}: reward{rewards.mean():.3f}(max:{rewards.max()}, min{rewards.min()})")

    def plot_rewards(self):
        indices = range(len(self.reward_log))
        means = np.array([rs.mean() for rs in self.reward_log])
        stds = np.array([rs.std() for rs in self.reward_log])
        plt.figure()
        plt.title("Reward History")
        plt.grid()
        plt.fill_between(indices, means - stds, means + stds, alpha=0.1, color="g")
        plt.plot(indices, means, "o-", color="g", label="reward")
        plt.legend(loc="best")
        plt.show()


# class Logger():
#     def __init__(self, log_dir="", dir_name=""):
#         self.log_dir = log_dir
#         if not log_dir:
#             self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
#         if not os.path.exists(self.log_dir):
#             os.mkdir(self.log_dir)
        
#         if dir_name:
#             self.log_dir = os.path.join(self.log_dir, dir_name)
#             if not os.path.exists(self.log_dir):
#                 os.mkdir(self.log_dir)
#         self._callback = tf.compat.v1.keras.callbacks.TensorBoard(self.log_dir)
    
#     @property
#     def writer(self):
#         return self._callback.writer
    
#     def set_model(self, model):
#         self._callback.set_model(model)
    
#     def path_of(self, file_name):
#         return os.path.join(self.log_dir, file_name)
    
#     def describe(self, name, values, episode=-1, step=-1):
#         mean = np.round(np.mean(values), 3)
#         std = np.round(np.std(values), 3)
#         desc = f"{name} is {mean} (+/-{std})"
#         if episode > 0:
#             print(f"At episode {episode}, {desc}")
#         elif step > 0:
#             print(f"At step {step}, {desc}")

#     def plot(self, name, values, interval=10):
#         indices = list(range(0, len(values), interval))
#         means = []
#         stds = []
#         for i in indices:
#             _values = values[i:(i + interval)]
#             means.append(np.mean(_values))
#             stds.append(np.std(_values))
#         means = np.array(means)
#         stds = np.array(stds)
#         plt.figure()
#         plt.title(f"{name} History")
#         plt.grid()
#         plt.fill_between(indices, means - stds, means + stds, alpha=0.1, color="g")
#         plt.plot(indices, means, "o-", color="g", 
#                 label=f"{name.lower()} per {interval} episode")
#         plt.legend(loc="best")
#         plt.show()
    
#     def write(self, index, name, value):
#         summary = tf.compat.v1.Summary()
#         summary_value = summary.value.add()
#         summary_value.tag = name
#         summary_value.simple_value = value
#         self.writer.add_summary(summary, index)
#         self.writer.flush()

#     def write_image(self, index, frames):

#         # Deal with a 'frames' as a list of sequential gray scaled image.
#         last_frames = [f[:, :, -1] for f in frames]
#         if np.min(last_frames[-1]) < 0:
#             scale = 127 / np.abs(last_frames[-1]).max()
#             offset = 128
#         else:
#             scale = 255 / np.max(last_frames[-1])
#             offset = 0
        
#         channel = 1  # gray scale
#         tag = f"frames_at_training_{index}"
#         values = []

#         for f in last_frames:
#             height, width = f.shape
#             array = np.asarray(f * scale + offset, dtype=np.unit8)
#             image = Image.fromarray(array)
#             output = io.BytesIO()
#             image.save(output, format="PNG")
#             image_string = output.getvalue()
#             output.close()
#             image = tf.compat.v1.Summary.Image(
#                 height=height, width=width, colorspace=channel,
#                 encoded_image_string=image_string)
#             value = tf.compat.v1.Summary.Value(tag=tag, image=image)
#             values.append(value)
        
#         summary = tf.compat.v1.Summary.Image(
#             height=height, width=width, colorspace=channel,
#             encoded_image_string=image_string)
#         value = tf.compat.v1.Summary.Value(tag=tag, image=image)
#         values.append(value)


def main(play):
    model_path = os.path.join(os.path.dirname(__file__), "ev_agent.h5")

    if play:
        env = EvolutionalTrainer.make_env()
        agent = EvolutionalAgent.load(env, model_path)
        agent.play(env, episode_count=5, render=True)
    else:
        trainer = EvolutionalTrainer()
        trained = trainer.train()
        trained.save(model_path)
        trainer.plot_rewards()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolutional Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")

    args = parser.parse_args()
    main(args.play)
