import sys
"""
from mcx import MCX

date = sys.argv[1]

mcx = MCX()
# mcx.run("configs/live/{}.json".format(date))
# mcx.run("configs/phantom.json")
"""

# from mcx_test import MCX
from mcx_test import MCX
mcx = MCX()
mcx.run("configs/phantom_muscle.json")
# mcx.run("configs/live/{}.json".format(sys.argv[1]))
