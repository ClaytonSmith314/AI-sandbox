
import torch
import numpy as np
import torch.nn as nn


def match(statement, variables, condition_statement):
    if condition_statement[0] == 'ALL':
        return True, variables
    if not len(statement)==len(condition_statement):
        return False, variables
    for i in range(len(condition_statement)):
        word = condition_statement[i]
        if isinstance(word, str):
            if word[0] == '$':
                if word in variables.keys():
                    if not variables[word]==statement[i]:
                        return False, variables
                else:
                    variables[word] = statement[i]
            else:
                if not word == statement[i]:
                    return False, variables
        else:
            if isinstance(word, tuple):
                if not isinstance(statement[i], tuple):
                    return False, variables
                substatement_valid, variables = match(statement[i], variables, word)
                if not substatement_valid:
                    return False, variables
            else:
                if not word == statement[i]:
                    return False, variables
    #print(pack(condition_statement)+' == '+pack(statement))
    return True, variables


def substitute(variables, statement, variable_symbol='$'):
    new_statement = list(statement)
    for i in range(len(statement)):
        word = statement[i]
        if isinstance(word, str):
            if word[0] == variable_symbol:
                if word in variables.keys():
                    new_statement[i] = variables[word]
        else:
            if isinstance(word, tuple):
                 new_statement[i] = substitute(variables, word)
    return tuple(new_statement)





class Rule:
    def __init__(self, special_literals, tensor_variables, functions,
                n_global_literals=8, n_variables = 8,
                placsticity=10, max_condition_len=3, max_string_len=6):
        self.conditions = []
        self.output = None

        self.plasticity = torch.tensor(placsticity) # measures how certain the model is in picking the largest value

        self.max_condition_len = max_condition_len
        self.max_string_len = max_string_len

        self.condition_prob_vals = np.random.standard_normal(size=(max_condition_len, max_string_len, len(token_bag)))
        self.assert_prob_vals = np.random.standard_normal(size=(max_string_len, len(token_bag)))

        # setup token bag
        self.special_literals = special_literals
        self.tensor_variables = tensor_variables
        self.functions = functions
        self.global_literals = []
        for i in range(n_global_literals):
            self.global_literals[i] = 'g' + str(i)
        self.variables = []
        for i in range(n_variables):
            self.variables[i] = '$'+str(i)

        self.word_bag = self.special_literals + \
                        self.tensor_variables + \
                        self.global_literals + \
                        self.variables

        self.condition_prob_vals = np.random.standard_normal(size=(max_condition_len, max_string_len, len(self.word_bag)))
        self.assert_prob_vals = np.random.standard_normal(size=(max_string_len, len(self.word_bag)))

        self.regenerate()


    def regenerate(self):
        # randomly selects the symbols (including
        if self.plasticity > 0: #if hardness is less than 1, then the rule is maliable. otherwise, not.
            self.conditions = []
            self.output = []
            for n in range(self.max_condition_len):
                self.conditions[n] = []
                for i in range(self.max_string_len):
                    exp_vals = np.exp(self.condition_prob_vals[n][i]/self.plasticity)
                    probs = exp_vals/np.sum(exp_vals)  # gets softmax distribution for symbol probs
                    choice = np.random.choice(probs)
                    self.conditions[n][i] = self.word_bag[choice]
            for i in range(self.max_string_len):
                exp_vals = np.exp(self.assert_prob_vals[i]/self.plasticity)
                probs = exp_vals/np.sum(exp_vals)
                choice = np.random.choice(probs)
                self.output[i] = (self.word_bag+self.functions)[choice]










class Derivation:
    def __init__(self, rule, source):
        self.rule = rule
        self.source = source


class KnowledgeManager:
    def __init__(self):
        self.knowledge = {}

    def addStatement(self, statement, rule, source):
        if not statement in self.knowledge.keys():
            self.knowledge[statement] = []
        self.knowledge[statement].append(Derivation(rule, source))

    def backward(self, statement, descrete_loss, continuous_loss):
        derivation = self.knowledge[statement]


