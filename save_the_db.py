from MySQLdb import connect
import pandas as pd 
import numpy as np 

from utils.mch import MCHHandler

mch = MCHHandler()
conn = connect(
    host="140.112.174.26",
    db="ijv",
    user="md703",
    passwd="MD703"
)

df = pd.read_sql("SELECT * FROM ijv_ann_3", con=conn)

for i, d in df.iterrows():
    
    mua = np.array([[d["skin_mua"], d["fat_mua"], d["muscle_mua"], d["ijv_mua"], d["cca_mua"]]]).T
    idx = d["idx"]
    try:
        s = mch.run_wmc_train("/home/md703/ijv/train/mch/{}.mch".format(idx), mua)
        
    except:
        continue
    print(idx, s)
    
    cursor = conn.cursor()
    qry = "UPDATE ijv_ann_3 SET reflectance_20 = '{:.4e}', ".format(s[0][0])
    qry += "reflectance_24 = '{:.4e}', reflectance_28 = '{:.4e}' ".format(s[0][1], s[0][2])
    qry += "WHERE idx='{}' AND abs(skin_mua-{})<=1e-6 AND ".format(idx, mua[0][0])
    qry += "abs(fat_mua-{})<=1e-6 AND abs(muscle_mua-{})<=1e-6 AND ".format(mua[1][0], mua[2][0])
    qry += "abs(ijv_mua-{})<=1e-6 AND abs(cca_mua-{})<=1e-6".format(mua[3][0], mua[4][0])
    cursor.execute(qry)
    
    conn.commit()