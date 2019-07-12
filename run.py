import sys
"""
from mcx import MCX

date = sys.argv[1]

mcx = MCX()
# mcx.run("configs/live/{}.json".format(date))
# mcx.run("configs/phantom.json")
"""

# from mcx_test import MCX
from mcx import MCX
mcx = MCX()

# mcx.run("configs/phantom.json")

# mcx.run("configs/quicktest.json")
mcx.run("configs/test_cv.json")
# mcx.run("configs/ijv_phantom_1.json")
# mcx.run("configs/ijv_phantom_2.json")
# mcx.run("configs/ijv_phantom_3.json")

# mcx.run("configs/phantom_muscle.json")

# mcx.run("configs/live/{}.json".format(sys.argv[1]))

#mcx.run("configs/live/20190507_min_no_prism.json")
#mcx.run("configs/live/20190507_max_no_prism.json")
#mcx.run("configs/live/20190502_min_no_prism.json")
#mcx.run("configs/live/20190502_max_no_prism.json")
#mcx.run("configs/live/20190510_min_no_prism.json")
#mcx.run("configs/live/20190510_max_no_prism.json")
#mcx.run("configs/live/20190511_min_no_prism.json")
#mcx.run("configs/live/20190511_max_no_prism.json")
