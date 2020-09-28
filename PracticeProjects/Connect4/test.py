import math

CRITICAL_Z = 1.96
EVAL_COUNT = 50
marginal_error = CRITICAL_Z * math.sqrt(.25 / EVAL_COUNT)
print('SAMPLE_INDUCED_MARGINAL_ERROR: +/- ' + str(round(100 * marginal_error, 1)) + '%\n\t*at 95% confidence')

