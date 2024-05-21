import util, submission
print('get_or_variable example:')
csp = util.create_or_csp()
alg = submission.BacktrackingSearch()
alg.solve(csp)
print('One of the optimal assignments:',  alg.optimalAssignment)

# print('\nMap coloring example:')
# csp = util.create_map_coloring_csp()
# alg = submission.BacktrackingSearch()
# alg.solve(csp)
# print('One of the optimal assignments:',  alg.optimalAssignment)

# print('\nWeighted CSP example:')
# csp = util.create_weighted_csp()
# alg = submission.BacktrackingSearch()
# alg.solve(csp)
# print('One of the optimal assignments:',  alg.optimalAssignment)
