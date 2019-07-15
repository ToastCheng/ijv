import numpy as np 
import matplotlib.pyplot as plt 
from engine import Engine 
from random import uniform, sample, shuffle

# def normalize(x):
#     return (x - x.mean())/x.std()

def normalize(x):
    x = x - x.min()
    x /= x.max()
    return x


class GA:

    def __init__(self, gene_size, para_size, target_max, target_min, geo_max, geo_min, ratio=0.05, iteration=1000, phantom=False, ink_ratio=None, dual_constraint=True):

        self.gene_size = gene_size
        self.para_size = para_size
        self.ratio = ratio
        self.iteration = iteration
        self.phantom = phantom
        self.ink_ratio = ink_ratio
        self.dual_constraint = dual_constraint

        self.para_range = [np.array([])
                           ]
        self.wavelength = np.arange(660, 811, 10)

        self.gene = []

        self.engine = Engine("train/model/20190710_001.pt", phantom=phantom)
        self.engine.wl = self.wavelength

        self.target_max = target_max
        self.target_min = target_min

        self.geo_max = geo_max
        self.geo_min = geo_min

    def generate(self, num_gene):
        for n in range(num_gene):
            args_max, args_min = self.get_args(self.geo_max, self.geo_min)
            gene = self.args2vec(args_max)

            spec_min = self.engine.get_spectrum(args_max)
            spec_max = self.engine.get_spectrum(args_min)


            abs_fit = normalize((np.log(spec_max) - np.log(spec_min))[6:])
            abs_target = normalize((np.log(self.target_max) - np.log(self.target_min))[6:])
            #error = (
            #    self.rmse(spec_max, self.target_max)+\
            #    self.rmse(spec_min, self.target_min)+\
            #    self.rmse(abs_fit, abs_target))/3
            if self.dual_constraint:
                error = (self.rmse(spec_max, self.target_max)+self.rmse(spec_min, self.target_min))/2  + self.rmse(abs_fit, abs_target) * 0.05
            else:
                error = (self.rmse(spec_max, self.target_max)+self.rmse(spec_min, self.target_min))/2 

            self.gene.append((gene, error))

        self.gene.sort(key=lambda x: x[1])

    def survive(self):

        num_pick = 100
        self.gene = self.gene[:100]
        self.generate(self.gene_size-100)
        self.gene.sort(key=lambda x: x[1])

        return self.gene[0:num_pick].copy()

    def crossover(self, good_gene, num=1):

        rand = np.random.rand()
        new_genes = []
        for n in range(num):
            gene = sample(good_gene, 2)

            if rand < 0.33:

                # linear crossover
                new_gene = gene[0] * 0.7 + gene[1] * 0.3
                # new_gene = self.bound(new_gene)

            elif rand < 0.66:

                new_gene = gene[0] * 1.5 - gene[1] * 0.5
                # new_gene = self.bound(new_gene)

            else:

                # exchange gene
                temp = [i for i in range(self.para_size)]
                shuffle(temp)
                new_gene = gene[1].copy()

                for i in temp[:self.para_size//2]:
                    new_gene[i] = gene[0][i].copy()

            # new_gene = self.bound(new_gene)
            new_gene = np.clip(new_gene, 0, None)
            new_genes += [new_gene]
        return new_genes

    def mutate(self):

        num_pick = np.ceil(len(self.gene) * self.ratio)
        mutate_list = np.random.randint(len(self.gene), size=int(num_pick))

        for i in mutate_list:

            # gene = (1.5 - np.random.rand()) * self.gene[i][0]
            temp = [x for x in range(self.para_size)]
            shuffle(temp)
            gene = self.gene[i][0].copy()

            for y in temp[:self.para_size//2]:
#                 gene[y] = (2-2*np.random.rand()) * gene[y]
                gene[y] = 100

            gene = np.clip(gene, 0, None)
            # gene = self.bound(gene)
            ####
            ####
            args_max = self.vec2args(gene, self.geo_max)
            args_min = self.vec2args(gene, self.geo_min)

            spec_max = self.engine.get_spectrum(args_min)
            spec_min = self.engine.get_spectrum(args_max)

            abs_fit = normalize((np.log(spec_max) - np.log(spec_min))[6:])
            abs_target = normalize((np.log(self.target_max) - np.log(self.target_min))[6:])
            #error = (
            #    self.rmse(spec_max, self.target_max)+\
            #    self.rmse(spec_min, self.target_min)+\
            #    self.rmse(abs_fit, abs_target))/3
            if self.dual_constraint:
                error = (self.rmse(spec_max, self.target_max)+self.rmse(spec_min, self.target_min))/2  + self.rmse(abs_fit, abs_target) * 0.05
            else:
                error = (self.rmse(spec_max, self.target_max)+self.rmse(spec_min, self.target_min))/2 
            self.gene += [(gene, error)]
            self.gene.sort(key=lambda x: x[1])
            del self.gene[-1]
            

    def bound(self, gene):

        upper = self.upper_bound
        lower = self.lower_bound

        return np.clip(gene, lower, upper)

    @staticmethod
    def rmse(x, y):

        x, y = np.array(x), np.array(y)
        return 100 * np.sqrt(((x-y)**2)).mean()/y.mean()

    def __call__(self, plot=False, verbose=False):

        self.gene = []

        self.generate(self.gene_size)

        rmse_list = []  # for plotting
        good_gene = None


        for epoch in range(self.iteration):

            rand = np.random.rand()
            if epoch % 100 == 0:
                good = self.survive()
            new_genes = self.crossover([i[0] for i in good].copy(), 50)
            
            for new_gene in new_genes:
                # calculate rmse
                args_max = self.vec2args(new_gene, self.geo_max)
                args_min = self.vec2args(new_gene, self.geo_min)

                spec_max = self.engine.get_spectrum(args_min)
                spec_min = self.engine.get_spectrum(args_max)

                abs_fit = normalize((np.log(spec_max) - np.log(spec_min))[6:])
                abs_target = normalize((np.log(self.target_max) - np.log(self.target_min))[6:])

                #error = (
                #    self.rmse(spec_max, self.target_max)+\
                #    self.rmse(spec_min, self.target_min)+\
                #    self.rmse(abs_fit, abs_target))/3
            if self.dual_constraint:
                error = (self.rmse(spec_max, self.target_max)+self.rmse(spec_min, self.target_min))/2  + self.rmse(abs_fit, abs_target) * 0.05
            else:
                error = (self.rmse(spec_max, self.target_max)+self.rmse(spec_min, self.target_min))/2 
                self.gene.append((new_gene, error))

            self.gene.sort(key=lambda x: x[1])
            
            rmse_list.append(self.gene[0][1])
            
            if rand < 0.3:
                
                self.mutate()

            if epoch % 50 == 0:

                if verbose:

                    print('epoch %d:' % epoch)
                    print('Size of gene library: %d' % len(self.gene))
                    print('best fit error: %f' % rmse_list[-1])
                    # print('gene: ')
                    # print(self.gene[0][0])

                if plot:
                    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
                    args_max_ = self.vec2args(self.gene[0][0], self.geo_max)
                    args_min_ = self.vec2args(self.gene[0][0], self.geo_min)
                    spec_max_ = self.engine.get_spectrum(args_min_)
                    spec_min_ = self.engine.get_spectrum(args_max_)
                    
                    ax[0].plot(self.wavelength, spec_max_, '--', label="fitting")
                    ax[0].plot(self.wavelength, spec_min_, '--', label="")
                    ax[0].plot(self.wavelength, self.target_max, label="measured")
                    ax[0].plot(self.wavelength, self.target_min, label="")
                    
                    ax[0].set_xlabel("wavelength [nm]")
                    ax[0].set_ylabel("reflectance [-]")
                    ax[0].grid()
                    ax[0].set_title("spectrum")
                    ax[0].legend()
                    
                    ax[1].plot(self.wavelength[6:], normalize((np.log(spec_max_) - np.log(spec_min_))[6:]), label="fitting")
                    ax[1].plot(self.wavelength[6:], normalize((np.log(self.target_max) - np.log(self.target_min))[6:]),  label="measured")
                    ax[1].set_xlabel("wavelength [nm]")
                    ax[1].set_ylabel("log(max/min) [-]")
                    ax[1].grid()
                    ax[1].legend()
                    ax[1].set_title("absorption")

                    ax[2].plot(np.arange(epoch+1),rmse_list,'C3')
                    ax[2].set_ylabel("loss")
                    ax[2].set_xlabel("iteration")
                    ax[2].grid()
                    ax[2].set_title('RMS percentage')
                    plt.show()




    @staticmethod
    def args2vec(args):
        vec = [
            args["skin"]["blood_volume_fraction"], 
            args["skin"]["ScvO2"], 
            args["skin"]["water_volume"], 
            args["skin"]["fat_volume"], 
            args["skin"]["melanin_volume"], 
            args["skin"]["muspx"], 
            args["skin"]["bmie"], 
            args["fat"]["blood_volume_fraction"], 
            args["fat"]["ScvO2"], 
            args["fat"]["fat_volume"], 
            args["fat"]["muspx"], 
            args["fat"]["bmie"],
            args["muscle"]["blood_volume_fraction"], 
            args["muscle"]["ScvO2"], 
            args["muscle"]["water_volume"], 
            args["muscle"]["collagen_volume"], 
            args["muscle"]["muspx"], 
            args["muscle"]["bmie"],
            args["IJV"]["ScvO2"], 
            args["IJV"]["muspx"], 
            args["IJV"]["bmie"],
            args["CCA"]["ScvO2"], 
            args["CCA"]["muspx"], 
            args["CCA"]["bmie"]
        ]
        return np.array(vec)

    @staticmethod
    def vec2args(vec, geo):
        args = {
            "skin":{
                "blood_volume_fraction": vec[0],
                "ScvO2": vec[1],
                "water_volume": vec[2],
                "fat_volume": vec[3],
                "melanin_volume": vec[4],
                "collagen_volume": 0,
                "n": 1.40,
                "g": 0.71,
                "muspx": vec[5],
                "bmie": vec[6]
            },

            "fat":{
                "blood_volume_fraction": vec[7],
                "ScvO2": vec[8],
                "water_volume": 0,
                "fat_volume": vec[9],
                "melanin_volume": 0,
                "collagen_volume": 0,
                "n": 1.40,
                "g": 0.90,
                "muspx": vec[10],
                "bmie": vec[11]
            },

            "muscle":{
                "blood_volume_fraction": vec[12],
                "ScvO2": vec[13],
                "water_volume": vec[14],
                "fat_volume": 0,
                "melanin_volume": 0,
                "collagen_volume": vec[15],
                "n": 1.40,
                "g": 0.9,
                "muspx": vec[16],
                "bmie": vec[17]
            },

            "IJV":{
                "blood_volume_fraction": 1.0,
                "ScvO2": vec[18],
                "water_volume": 0,
                "fat_volume": 0,
                "melanin_volume": 0,
                "collagen_volume": 0,
                "n": 1.40,
                "g": 0.90,
                "muspx": vec[19],
                "bmie": vec[20]
            },

            "CCA":{
                "blood_volume_fraction": 1.0,
                "ScvO2": vec[21],
                "water_volume": 0,
                "fat_volume": 0,
                "melanin_volume": 0,
                "collagen_volume": 0,
                "n": 1.40,
                "g": 0.94,
                "muspx": vec[22],
                "bmie": vec[23]
            },
            "geometry": geo
        }
        return args

    def get_args(self, geo_max, geo_min):
        if not self.phantom:
            x_range = {
                "sb": (0, 0.1),
                "ss": (0.5, 1.0),
                "sw": (0, 1.0),
                "sf": (0, 1.0),
                "sm": (0, 1.0),

                "fb": (0, 0.1),
                "fs": (0, 1.0),
                "ff": (0.9, 1),

                "mb": (0.005, 0.1),
                "ms": (0.0, 1.0),
                "mw": (0.0, 1.0),

                "is": (0.4, 0.8),

                "cs": (0.85, 1.0),
            }
        else:
            x_range = {
                "sb": (0, 0.1),
                "ss": (0.5, 1.0),
                "sw": (0, 1.0),
                "sf": (0, 1.0),
                "sm": (0, 1.0),

                "fb": (0, 0.1),
                "fs": (0, 1.0),
                "ff": (0.9, 1),

                "mb": (0.005, 0.1),
                "ms": (0.0, 1.0),
                "mw": (0.0, 1.0),

                "is": (self.ink_ratio-0.1, self.ink_ratio+0.1),

                "cs": (0.85, 1.0),
            }


        # absorption
        sb = uniform(x_range["sb"][0], x_range["sb"][1])
        ss = uniform(x_range["ss"][0], x_range["ss"][1])
        sw = uniform(x_range["sw"][0], max(x_range["sw"][0], min(x_range["sw"][1], 1-sb)))
        sf = uniform(x_range["sf"][0], max(x_range["sf"][0], min(x_range["sf"][1], 1-sb-sw)))
        sm = 1-sb-sw-sf
        
        fb = uniform(x_range["fb"][0], x_range["fb"][1])
        fs = uniform(x_range["fs"][0], x_range["fs"][1])
        ff = 1 - fb
        
        mb = uniform(x_range["mb"][0], x_range["mb"][1])
        ms = uniform(x_range["ms"][0], x_range["ms"][1])
        mw = uniform(x_range["mw"][0], max(x_range["mw"][0], min(x_range["mw"][1], 1-mb)))
        mc = 1 - mb - mw
        is_ = uniform(x_range["is"][0], x_range["is"][1])
        cs = uniform(x_range["cs"][0], x_range["cs"][1])
        
        # scattering
        s_musp = uniform(29.7, 48.9)
        s_bmie = uniform(0.705, 2.453)
        
        f_musp = uniform(13.7, 35.8)
        f_bmie = uniform(0.385, 0.988)
        
        m_musp = uniform(9.8, 13.0)
        m_bmie = uniform(0.926, 2.82)
        
        i_musp = uniform(0.1, 1)
        i_bmie = uniform(1, 1)
        
        c_musp = uniform(0.1, 1)
        c_bmie = uniform(1, 1)        
        
        args_max = {
            "skin":{
                "blood_volume_fraction": sb,
                "ScvO2": ss,
                "water_volume": sw,
                "fat_volume": sf,
                "melanin_volume": sm,
                "collagen_volume": 0,
                "n": 1.40,
                "g": 0.71,
                "muspx": s_musp,
                "bmie": s_bmie
            },

            "fat":{
                "blood_volume_fraction": fb,
                "ScvO2": fs,
                "water_volume": 0,
                "fat_volume": ff,
                "melanin_volume": 0,
                "collagen_volume": 0,
                "n": 1.40,
                "g": 0.90,
                "muspx": f_musp,
                "bmie": f_bmie
            },

            "muscle":{
                "blood_volume_fraction": mb,
                "ScvO2": ms,
                "water_volume": mw,
                "fat_volume": 0,
                "melanin_volume": 0,
                "collagen_volume": mc,
                "n": 1.40,
                "g": 0.9,
                "muspx": m_musp,
                "bmie": m_bmie
            },

            "IJV":{
                "blood_volume_fraction": 1.0,
                "ScvO2": is_,
                "water_volume": 0,
                "fat_volume": 0,
                "melanin_volume": 0,
                "collagen_volume": 0,
                "n": 1.40,
                "g": 0.90,
                "muspx": i_musp,
                "bmie": i_bmie
            },

            "CCA":{
                "blood_volume_fraction": 1.0,
                "ScvO2": cs,
                "water_volume": 0,
                "fat_volume": 0,
                "melanin_volume": 0,
                "collagen_volume": 0,
                "n": 1.40,
                "g": 0.94,
                "muspx": c_musp,
                "bmie": c_bmie
            },
            "geometry": geo_max
        }
        args_min = {
            "skin":{
                "blood_volume_fraction": sb,
                "ScvO2": ss,
                "water_volume": sw,
                "fat_volume": sf,
                "melanin_volume": sm,
                "collagen_volume": 0,
                "n": 1.40,
                "g": 0.71,
                "muspx": s_musp,
                "bmie": s_bmie
            },

            "fat":{
                "blood_volume_fraction": fb,
                "ScvO2": fs,
                "water_volume": 0,
                "fat_volume": ff,
                "melanin_volume": 0,
                "collagen_volume": 0,
                "n": 1.40,
                "g": 0.90,
                "muspx": f_musp,
                "bmie": f_bmie
            },

            "muscle":{
                "blood_volume_fraction": mb,
                "ScvO2": ms,
                "water_volume": mw,
                "fat_volume": 0,
                "melanin_volume": 0,
                "collagen_volume": mc,
                "n": 1.40,
                "g": 0.9,
                "muspx": m_musp,
                "bmie": m_bmie
            },

            "IJV":{
                "blood_volume_fraction": 1.0,
                "ScvO2": is_,
                "water_volume": 0,
                "fat_volume": 0,
                "melanin_volume": 0,
                "collagen_volume": 0,
                "n": 1.40,
                "g": 0.90,
                "muspx": i_musp,
                "bmie": i_bmie
            },

            "CCA":{
                "blood_volume_fraction": 1.0,
                "ScvO2": cs,
                "water_volume": 0,
                "fat_volume": 0,
                "melanin_volume": 0,
                "collagen_volume": 0,
                "n": 1.40,
                "g": 0.94,
                "muspx": c_musp,
                "bmie": c_bmie
            },
            "geometry": geo_min
        }
        return args_max, args_min

