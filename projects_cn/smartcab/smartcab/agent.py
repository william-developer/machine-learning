import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.q = {}
	self.actions=env.valid_actions
	self.alpha=0.4
	self.gamma=0.1
	self.epsilon =0.1 
	self.sucess=0;
	self.testtimes=0;

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
	self.testtimes+=1
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
	state=(self.next_waypoint,inputs['light'],inputs['oncoming'],inputs['left'])
	q = [self.q.get((state, a),0.0) for a in self.actions]
        maxQ = max(q)
	if random.random() < self.epsilon:
		action=random.choice(self.actions)
	else:
		count = q.count(maxQ)
       		if count > 1:
            		best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            		i = random.choice(best)
        	else:
            		i = q.index(maxQ)
		action = self.actions[i]
	reward=self.env.act(self,action)

	new_inputs = self.env.sense(self)
	next_state=(self.planner.next_waypoint(),new_inputs['light'],new_inputs['oncoming'],new_inputs['left'])
	
	self.q[(state,action)]=(1-self.alpha)*self.q.get((state,action),0.0)+self.alpha*(reward+self.gamma*max([self.q.get((next_state,a),0.0) for a in self.actions]))

	if self.env.done :
		self.sucess+=1	
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
	if deadline<=0 or self.env.done:
		print self.sucess,self.testtimes

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
