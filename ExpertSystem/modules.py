
from copy import deepcopy

# things we need
# 1. rules of inference. each rule has a "foreach" condition
# 2.


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



special_symbols = '+=-/%^*~|'

def unpack(text, last=0):
    statement = []
    i=last
    while i < len(text):
        if text[i] in special_symbols:
            if not last==i:
                statement.append(text[last:i])
            statement.append(text[i])
            last=i+1
        if text[i] == ')':
            if not last==i:
                statement.append(text[last:i])
            return tuple(statement), i
        if text[i] == ' ' or text[i]==',':
            if not last==i:
                statement.append(text[last:i])
            last=i+1
        if text[i] == '(':
            if not last==i:
                statement.append(text[last:i])
            last=i+1
            sub_statement, i = unpack(text, last)
            last=i+1
            statement.append(sub_statement)
        i += 1
    if not last == i:
        statement.append(text[last:i])
    return tuple(statement)

def pack(statement, delimeter=' '):
    text = ''
    for word in statement:
        if isinstance(word, tuple):
            text += '('+pack(word, delimeter)+')' + delimeter
        else:
            text += str(word)+delimeter
    return text[:-1]


def run_forward_query(query_, axoims_, rules_, max_repetitions=100, print_=True, stop_at_first=False):
    results_ = []
    variable_assignments = []
    theorems = []
    n=0
    print('---Axoim Matches---')
    for statement_ in axoims_:
        ismatch, variables = match(statement_, {}, query_)
        if ismatch:
            n += 1
            results_.append(statement_)
            variable_assignments.append(variables)
            if print_:
                print(f'{n}:\t'+pack(statement_))
            if stop_at_first:
                return results_, variable_assignments
    print('---Theorem Matches---')
    for _ in range(max_repetitions):
        no_theorems_added = True
        rule_number = 0
        for rule in rules_:
            rule_number += 1
            #print(f'\n--Rule {rule_number}--')
            new_theorems = rule.forward_test(axoims_+theorems)
            theorems += new_theorems
            if len(new_theorems) > 0:
                no_theorems_added = False
            for statement_ in new_theorems:
                ismatch, variables = match(statement_, {}, query_)
                if ismatch:
                    n += 1
                    results_.append(statement_)
                    variable_assignments.append(variables)
                    if print_:
                        print(f'{n}:\tRule({rule_number})\t' + pack(statement_))
                    if stop_at_first:
                        return results_, variable_assignments
        if no_theorems_added:
            break
    return results_, variable_assignments





class Rule:
    def __init__(self, conditions, result, unequal_variables=()):
        if isinstance(conditions[0], str):
            str_conditions = conditions
            condition_statements = []
            for i in range(len(str_conditions)):
                condition_statements.append(unpack(str_conditions[i]))
            self.condition = lambda _index, _statement, _variables: match(_statement, _variables, condition_statements[_index])
        else:
            self.condition = lambda _index, _statement, _variables: conditions[_index](_statement, _variables)

        if isinstance(result, str):
            result_statement = unpack(result)
            result = lambda _variables: substitute(_variables, result_statement)
        self.result = result
        self.conditions_len = len(conditions)
        self.unequal_vars = unequal_variables

    def forward_test(self, statements, variables={}, n=0, include_old_asserts=False,
                     batch=None):

        asserts = []
        for statement in statements:
            valid, new_variables = self.condition(n, statement, deepcopy(variables))
            if valid:
                #print('^^^-'+str(n))
                if n+1==self.conditions_len:
                    stopassert = False
                    for i in range(len(self.unequal_vars)):
                        for j in range(i+1,len(self.unequal_vars)):
                            if new_variables[self.unequal_vars[i]] == new_variables[self.unequal_vars[j]]:
                                stopassert = True
                    result = self.result(new_variables)
                    if result in statements+asserts and not include_old_asserts:
                        stopassert = True
                    if not stopassert:
                        asserts.append(result)
                else:
                    new_asserts = self.forward_test(statements, new_variables, n+1)
                    for new_assert in new_asserts:
                        if not new_assert in asserts:
                            asserts.append(new_assert)
        return asserts







