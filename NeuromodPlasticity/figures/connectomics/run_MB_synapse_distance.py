import numpy as np
import pandas as pd
import scipy as sp
import pathlib
import os
from multiprocessing import Pool
import cloudpickle
import pickle

import seaborn as sns
from matplotlib import pyplot as plt
import networkx as nx

import navis 
import navis.interfaces.neuprint as neu
import neuprint as neu_orig
import NeuromodPlasticity as nmp 


def get_kc_syn():

    kcs, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type='KCg.*'))
    mbons, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type='MBON.*', 
                                                            inputRois=['gL(L)', 'gL(R)'], roi_req='any'))

    dans, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type=['PPL.*', 'PAM.*'],
                                                        outputRois = ['gL(L)', 'gL(R)'], roi_req='any'))                                               


    kc_mbon_synapses = neu_orig.fetch_synapse_connections(kcs['bodyId'],
                                                        mbons['bodyId'],
                                                        neu_orig.SynapseCriteria(rois=['gL(L)', 'gL(R)'])
                                                        ) 

    dan_kc_synapses = neu_orig.fetch_synapse_connections(dans['bodyId'],
                                                        kcs['bodyId'],
                                                        neu_orig.SynapseCriteria(rois=['gL(L)', 'gL(R)'])
                                                        )
    
    kc_mbon_synapses['type'] = 'mbon_pre'
    kc_mbon_synapses['x'], kc_mbon_synapses['y'], kc_mbon_synapses['z'] = kc_mbon_synapses['x_pre'], kc_mbon_synapses['y_pre'], kc_mbon_synapses['z_pre']
    dan_kc_synapses['type'] = 'dan_post'
    dan_kc_synapses['x'], dan_kc_synapses['y'], dan_kc_synapses['z'] = dan_kc_synapses['x_post'], dan_kc_synapses['y_post'], dan_kc_synapses['z_post']

    kc_syns = pd.concat([kc_mbon_synapses, dan_kc_synapses], ignore_index=True)
    return kc_syns, kcs


def calc_syn_distance(args):

    kc_bodyId, cell_syns = args[0], args[1]

    kc_skel = neu_orig.fetch_skeleton(kc_bodyId, heal=True, with_distances=True)
    kc_skel = neu_orig.upsample_skeleton(kc_skel, 1)
    kc_skel = neu_orig.attach_synapses_to_skeleton(kc_skel, cell_syns)

    # get info to add to nodes
    struct_dict = {}
    for _, row in kc_skel.iterrows():
        struct_dict[row['rowId']]=row['structure']

    # convert skeleton to networkx graph and add info to nodes
    kc_nx = neu_orig.skeleton_df_to_nx(kc_skel, with_distances=True,directed=False)
    nx.set_node_attributes(kc_nx, struct_dict, 'structure')

    # get node names of mbon synapses
    pre_syns = kc_skel.loc[kc_skel['structure']=='mbon_pre']
    pre_nodes= [row['rowId'] for _, row in pre_syns.iterrows()]

    post_type = 'dan_post'
    # get node names of post_synapses
    post_syns = kc_skel.loc[kc_skel['structure']==post_type,:]

    distances, euc_distances = {}, {}
    distances[post_type] = np.zeros((len(pre_nodes),))
    euc_distances[post_type] = np.zeros((len(pre_nodes),))
    for i, pre_node in enumerate(pre_nodes):
        try:
            # filter post nodes to ones withing 10 microns euclidean distance
            pre_syn = kc_skel.loc[kc_skel['rowId']==pre_node,:]
            euc_dist = np.linalg.norm(pre_syn.loc[:,('x','y','z')]._values - post_syns.loc[:,('x','y','z')]._values, axis=1)
            euc_distances[post_type][i] = np.nanmin(euc_dist)
            post_syns_close = post_syns.loc[euc_dist<10*1000/8,:]

            post_nodes = []
            for _, row in post_syns_close.iterrows():
                post_nodes.append(row['rowId'])
            
            dist, path = nx.multi_source_dijkstra(kc_nx, post_nodes, target=pre_node, weight='distance', cutoff=15*1000/8)
            distances[post_type][i] = np.nanmin(dist)
        except:
            distances[post_type][i] = np.nan
            euc_distances[post_type][i] = np.nan

    return {'cable_distance': distances, 'euclidean_distance': euc_distances}


def run_kc_type():
    syns, nrns = get_kc_syn()

    args = []
    for id in nrns['bodyId'].tolist():
        mask = (syns['type']=='mbon_pre') * (syns['bodyId_pre']==id)
        mask = mask | ((syns['type'].str.contains('post')) * (syns['bodyId_post']==id))
        cell_syns = syns.loc[mask,:]
        args.append((id, cell_syns))
        

    with Pool(processes=8) as pool:
        kc_distances = pool.map(calc_syn_distance, args)

    with open(figfolder / 'kc_syn_distances.pkl', 'wb') as f:
        pickle.dump(kc_distances, f)

if __name__=="__main__":

    figfolder = pathlib.Path('/media/mplitt/SSD_storage/fig_scratch/EL_connectomics/syn_distances')
    figfolder.mkdir(parents=True, exist_ok=True)

    c = nmp.connectomics.npt_client()


    
    run_kc_type()
