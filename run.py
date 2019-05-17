import sys
from mcx import MCX

date = sys.argv[1]

mcx = MCX()
mcx.run("configs/live/{}.json".format(date))