import pygraphviz as pgv
import networkx as nx
import numpy as np
import re
import sys
import os

#TGT=args_.input_variable
#DOTNAME=args_.DOTNAME[0]
#MAPNAME = args_.map
#BIOME = args_.biome
#TIME_START = args_.time_start
#TIME_END = args_.time_end



variable_bin_map = np.load(MAPNAME, allow_pickle=True)
NMAP = variable_bin_map.item()
LABELS = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}

gv = pgv.AGraph(DOTNAME, strict=False, directed=True)
G = nx.DiGraph(gv)

labels=nx.get_node_attributes(G,'label')
edgelabels = nx.get_edge_attributes(G, "label")

N_hm=[]

def dequantize(letter, bin_arr):
    if letter is np.nan or letter == 'nan':
        return np.nan
    lo = LABELS[letter]
    hi = lo + 1
    val = (bin_arr[lo] + bin_arr[hi]) / 2
    return val

def deQuantizer(letter, biome_prefix, time_start=None, time_end=None):
    vals = []
    if time_start is None and time_end is None:
        # average over all
        for biome_key in NMAP:
            if biome_prefix in biome_key:
                vals.append(dequantize(letter, NMAP[biome_key]))
    elif time_start is None:
        time_end = int(time_end)
        for biome_key in NMAP:
            _, biome_full, time, _ = re.split(r'(.*)_(\d+)', biome_key)
            time = int(time)
            if time <= time_end:
                vals.append(dequantize(letter, NMAP[biome_key]))
    elif time_end is None:
        time_start = int(time_start)
        for biome_key in NMAP:
            _, biome_full, time, _ = re.split(r'(.*)_(\d+)', biome_key)
            time = int(time)
            if time >= time_start:
                vals.append(dequantize(letter, NMAP[biome_key]))
    else: # both present
        time_start = int(time_start)
        time_end = int(time_end)
        for biome_key in NMAP:
            _, biome_full, time, _ = re.split(r'(.*)_(\d+)', biome_key)
            time = int(time)
            if time_start <= time <= time_end:
                vals.append(dequantize(letter, NMAP[biome_key]))
    return np.mean(vals)

def getNum(D, bin_names):
    R={}
    for k in D:
        v = D[k]
        bin_name = bin_names[k]
        biome_prefix = bin_name.split('_')[0]
        R[k]=np.array([deQuantizer(str(x).strip(), biome_prefix, time_start=TIME_START, time_end=TIME_END) for x in v]).mean()
    return R

        
def downStream(nodeset):
    cLeaf=[x for x in nodeset if G.out_degree(x)==0 and G.in_degree(x)==1]
    oLabels={k:str(v.split('\n')[0]) for (k,v) in labels.items() if k in cLeaf}
    # k: prob, k+1: column name
    bin_names = {str(int(k) - 1): v.strip() for (k, v) in labels.items() 
                 if str(int(k) - 1) in cLeaf}

    frac={k:float(v.split('\n')[2].replace('Frac:','')) for (k,v) in labels.items() if k in cLeaf}
    SUM=np.array(frac.values()).sum()
    bin_names = {k: BIOME for k in oLabels}
    num_oLabels=getNum(oLabels, bin_names)
    res={}
    for (k,v) in frac.items():
        res[k]=(num_oLabels[k],v)
    return res,SUM


def getRF(i_):
    cNodes=list(nx.descendants(G,N_hm[i_]))
    nextNodes=nx.neighbors(G,N_hm[i_])
    nextedge={}
    edgeProp={}
    SUM=0.
    for nn in list(nextNodes):
        nextedge[nn]=[str(x) for x in edgelabels[(N_hm[i_],nn)].split('\\n')]
        
        res,s=downStream(list(nx.descendants(G,nn)))
        if len(list(nx.descendants(G,nn))) == 0:
            res,s=downStream([nn])
        
        edgeProp[nn]=res
        if sys.version_info[0] < 3:
            SUM=SUM+s
        else:
            SUM=SUM+list(s)[0]
        
        bin_names = {k: BIOME if len(labels[k].split('\n')) > 1 else labels[k] for k in nextedge}
        num_nextedge=getNum(nextedge, bin_names)

    for (k,v) in edgeProp.items():
        r=0
        for (kk,vv) in v.items():
            r=r+(vv[0]*vv[1])
        num_nextedge[k]=np.append(num_nextedge[k],r)
    RF=pd.DataFrame(num_nextedge)
    RF.index=['inputs_'+str(i_),'response_'+str(i_)]
    RF.columns=['x'+str(i) for i in np.arange(len(RF.columns))]
    return RF




for (k,v) in labels.items():
    if TGT in v:
        N_hm=N_hm+[k]

if len(N_hm)==0:
    print('Target not present in predictor : 0')
    quit()










import numpy as np
from quasinet import qnet


class Forecaster:
    """Forecast the data week by week by sequantially generating qnet predictions for the next timestamp and using the filled timestamp to update qnet predictions
    """

    def __init__(self, qnet_orchestrator):
        """Initialization
        Args:
            qnet_orchestrator (qbiome.QnetOrchestrator): an instance with a trained qnet model
        """
        self.qnet_orchestrator = qnet_orchestrator
        self.quantizer = qnet_orchestrator.quantizer

    def forecast_data(self, data, start_week, end_week, n_samples=100):
        """Forecast the data matrix from `start_week` to `end_week`
        Output format:
        |   subject_id | variable         |   week |    value |
        |-------------:|:-----------------|-------:|---------:|
        |            1 | Actinobacteriota |     27 | 0.36665  |
        |            1 | Bacteroidota     |     27 | 0.507248 |
        |            1 | Campilobacterota |     27 | 0.002032 |
        Args:
            data (numpy.ndarray): 2D array of label strings, produced by `self.get_qnet_inputs`
            start_week (int): start predicting from this week
            end_week (int): end predicting after this week
            n_samples (int, optional): the number of times to sample from qnet predictions for one masked entry. Defaults to 100.
        Returns:
            pandas.DataFrame: see format above
        """
        forecasted_matrix = np.empty(data.shape)
        for idx, seq in enumerate(data):
            forecasted_seq = self.qnet_orchestrator.predict_sequentially_by_week(
                seq, start_week, end_week, n_samples=n_samples
            )
            forecasted_matrix[idx] = forecasted_seq

        df = self.quantizer.add_meta_to_matrix(forecasted_matrix)
        # convert to plottable format
        plot_df = self.quantizer.melt_into_plot_format(df)
        return plot_df
