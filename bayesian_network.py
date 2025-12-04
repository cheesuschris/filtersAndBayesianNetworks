""" Bayesian networks """

from probability import BayesNet, enumeration_ask, elimination_ask, rejection_sampling, likelihood_weighting, gibbs_ask
from timeit import timeit, repeat
import pickle
import numpy as np

T, F = True, False

class DataPoint:
    """
    Represents a single datapoint gathered from one lap.
    Attributes are exactly the same as described in the project spec.
    """
    def __init__(self, muchfaster, early, overtake, crash, win):
        self.muchfaster = muchfaster
        self.early = early
        self.overtake = overtake
        self.crash = crash
        self.win = win

def generate_bayesnet():
    """
    Generates a BayesNet object representing the Bayesian network in Part 2
    returns the BayesNet object
    """
    bayes_net = BayesNet()
    # load the dataset, a list of DataPoint objects
    data = pickle.load(open("data/bn_data.p","rb"))
    # BEGIN_YOUR_CODE ######################################################
    
    total = len(data)
    muchfaster_count = 0
    early_count = 0
    for d in data:
        if d.muchfaster == T:
            muchfaster_count +=1 
        if d.early == T:
            early_count += 1
    p_muchfaster = muchfaster_count / total
    p_early = early_count / total
    
    overtake_conditional = {}
    for much_faster in [T, F]:
        for early in [T, F]:
            parent_count = 0
            overtake_count = 0
            for d in data:
                if d.muchfaster == much_faster and d.early == early:
                    parent_count += 1
                if d.muchfaster == much_faster and d.early == early and d.overtake == T:
                    overtake_count += 1
            if parent_count > 0:
                overtake_conditional[(much_faster, early)] = overtake_count / parent_count
            else:
                overtake_conditional[(much_faster, early)] = 0
    
    crash_conditional = {}
    for much_faster in [T, F]:
        for early in [T, F]:
            parent_count = 0
            crash_count = 0
            for d in data:
                if d.muchfaster == much_faster and d.early == early:
                    parent_count += 1
                if d.muchfaster == much_faster and d.early == early and d.crash == T:
                    crash_count += 1
            if parent_count > 0:
                crash_conditional[(much_faster, early)] = crash_count / parent_count
            else:
                crash_conditional[much_faster, early] = 0
    
    win_conditional = {}
    for overtake in [T, F]:
        for crash in [T, F]:
            parent_count = 0
            win_count = 0
            for d in data:
                if d.overtake == overtake and d.crash == crash:
                    parent_count += 1
                if d.overtake == overtake and d.crash == crash and d.win == T:
                    win_count += 1
            if parent_count > 0:
                win_conditional[(overtake, crash)] = win_count / parent_count
            else:
                win_conditional[(overtake, crash)] = 0
    
    bayes_net = BayesNet([
        ('MuchFaster', '', p_muchfaster),
        ('Early', '', p_early),
        ('Overtake', 'MuchFaster Early', overtake_conditional),
        ('Crash', 'MuchFaster Early', crash_conditional),
        ('Win', 'Overtake Crash', win_conditional)
    ])
    
    # END_YOUR_CODE ########################################################
    return bayes_net

def find_best_overtake_condition(bayes_net):
    """
    Finds the optimal condition for overtaking the car, as described in Part 3
    Returns the optimal values for (MuchFaster,Early)
    """
    # BEGIN_YOUR_CODE ######################################################

    #Mathematically:
    #P(Win = T, Crash = F | MuchFaster, Early) = P(Win = T | Crash = F, MuchFaster, Early) * P(Crash = F | MuchFaster, Early)
    #So we need:
    #P(Crash = F | MuchFaster = much_faster, Early = early)
    #P(Win = T | Crash = F, MuchFaster = much_faster, Early = early)
    p_best = -1
    much_faster_best, early_best = (None, None)
    for much_faster in [T, F]:
        for early in [T, F]:
            crash_probability_distribution = elimination_ask('Crash', dict(MuchFaster=much_faster, Early=early), bayes_net)
            p_crash_false = crash_probability_distribution[F]
            win_probability_distribution = elimination_ask('Win', dict(Crash=F, MuchFaster=much_faster, Early=early), bayes_net)
            p_win_given_crash_false = win_probability_distribution[T]
            p_win_without_crashing = p_win_given_crash_false * p_crash_false
            if p_win_without_crashing > p_best:
                p_best = p_win_without_crashing
                much_faster_best, early_best = much_faster, early
    return (much_faster_best, early_best)

    # END_YOUR_CODE ########################################################

def main():
    bayes_net = generate_bayesnet()
    cond = find_best_overtake_condition(bayes_net)
    print("Best overtaking condition: MuchFaster={}, Early={}".format(cond[0],cond[1]))

if __name__ == "__main__":
    main()

