
from ExpertSystem.modules import unpack, run_forward_query, pack
from ExpertSystem.relations_rules import rules, axoims

if __name__ == '__main__':
    for i in range(len(axoims)):
        axoims[i] = unpack(axoims[i])

    print('welcome to the Expert System!')
    action = ''
    while not action == 's':
        #print("choose action:\n\ta=add rule, q=query, e=explore, s=stop")
        action = 'q' #input('? ')
        if action == 'q':
            query = input('enter query: ')
            query_statement = unpack(query)
            results, variable_assign = run_forward_query(query_statement, axoims, rules, 500)
            print(f'result count: {len(results)}')

        if action == 'e':
            pass