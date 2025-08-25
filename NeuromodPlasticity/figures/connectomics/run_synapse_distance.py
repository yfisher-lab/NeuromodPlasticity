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

def get_er_syn(ertype = 'ER4d'):
    ers, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type = ertype))

    els, _= neu.queries.fetch_neurons(neu.NeuronCriteria(type = 'EL'))
    exr2s, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type='ExR2'))
    exr3s, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type='ExR3'))
    epgs, _= neu.queries.fetch_neurons(neu.NeuronCriteria(type = 'EPG'))


    #  get synapses from example ER4d to EPG and EL to example ER4d
    er_epg_synapses = neu_orig.fetch_synapse_connections(ers['bodyId'],
                                                        epgs['bodyId'], #example_epg_bodyID,
                                                        neu_orig.SynapseCriteria(rois='EB'))

    el_er_synapses = neu_orig.fetch_synapse_connections(els['bodyId'], #example_el_bodyId,
                                                        ers['bodyId'],
                                                        neu_orig.SynapseCriteria(rois='EB'))
    #exr2
    exr2_er_synapses = neu_orig.fetch_synapse_connections(exr2s['bodyId'],
                                                        ers['bodyId'],
                                                        neu_orig.SynapseCriteria(rois='EB'))

    #exr3
    exr3_er_synapses = neu_orig.fetch_synapse_connections(exr3s['bodyId'],
                                                            ers['bodyId'],
                                                            neu_orig.SynapseCriteria(rois='EB'))

    er_epg_synapses['type'] = 'epg_pre'
    er_epg_synapses['x'], er_epg_synapses['y'], er_epg_synapses['z'] = er_epg_synapses['x_pre'], er_epg_synapses['y_pre'], er_epg_synapses['z_pre']


    el_er_synapses['type'] = 'el_post'
    el_er_synapses['x'], el_er_synapses['y'], el_er_synapses['z'] = el_er_synapses['x_post'], el_er_synapses['y_post'], el_er_synapses['z_post']

    exr2_er_synapses['type'] = 'exr2_post'
    exr2_er_synapses['x'], exr2_er_synapses['y'], exr2_er_synapses['z'] = exr2_er_synapses['x_post'], exr2_er_synapses['y_post'], exr2_er_synapses['z_post']

    exr3_er_synapses['type'] = 'exr3_post'
    exr3_er_synapses['x'], exr3_er_synapses['y'], exr3_er_synapses['z'] = exr3_er_synapses['x_post'], exr3_er_synapses['y_post'], exr3_er_synapses['z_post']



    er_syns = pd.concat(( er_epg_synapses, el_er_synapses, exr2_er_synapses, exr3_er_synapses), ignore_index=True)
    return er_syns, ers

def calc_syn_distance(args, post_types = ['el_post', 'exr2_post', 'exr3_post']):
    er_bodyId, cell_syns = args[0], args[1]
    # get skeleton and attach synapses
    er_skel = neu_orig.fetch_skeleton(er_bodyId, heal=True, with_distances=True)
    er_skel = neu_orig.upsample_skeleton(er_skel, 1)
    er_skel = neu_orig.attach_synapses_to_skeleton(er_skel, cell_syns)

    # get info to add to nodes
    struct_dict = {}
    for _, row in er_skel.iterrows():
        struct_dict[row['rowId']]=row['structure']

    # convert skeleton to networkx graph and add info to nodes
    er_nx = neu_orig.skeleton_df_to_nx(er_skel, with_distances=True,directed=False)
    nx.set_node_attributes(er_nx, struct_dict, 'structure')

    # get node names of epg synapses
    pre_syns = er_skel.loc[er_skel['structure']=='epg_pre']
    pre_nodes= [row['rowId'] for _, row in pre_syns.iterrows()]

    distances, euc_distances = {}, {}
    for key in post_types:

        # get node names of post_synapses
        post_syns = er_skel.loc[er_skel['structure']==key,:]


        distances[key] = np.zeros((len(pre_nodes),))
        euc_distances[key] = np.zeros((len(pre_nodes),))
        for i, pre_node in enumerate(pre_nodes):
            try:
                # filter post nodes to ones withing 10 microns euclidean distance
                pre_syn = er_skel.loc[er_skel['rowId']==pre_node,:]
                euc_dist = np.linalg.norm(pre_syn.loc[:,('x','y','z')]._values - post_syns.loc[:,('x','y','z')]._values, axis=1)
                euc_distances[key][i] = np.nanmin(euc_dist)
                post_syns_close = post_syns.loc[euc_dist<10*1000/8,:]

                post_nodes = []
                for _, row in post_syns_close.iterrows():
                    post_nodes.append(row['rowId'])
                
                dist, path = nx.multi_source_dijkstra(er_nx, post_nodes, target=pre_node, weight='distance', cutoff=15*1000/8)
                distances[key][i] = np.nanmin(dist)
            except:
                distances[key][i] = np.nan
                euc_distances[key][i] = np.nan

    return {'cable_distance': distances, 'euclidean_distance': euc_distances}


def run_er_type(er_type):
    syns, nrns = get_er_syn(er_type)

    args = []
    for id in nrns['bodyId'].tolist():
        mask = (syns['type']=='epg_pre') * (syns['bodyId_pre']==id)
        mask = mask | ((syns['type'].str.contains('post')) * (syns['bodyId_post']==id))
        cell_syns = syns.loc[mask,:]
        args.append((id, cell_syns))
        

    with Pool(processes=8) as pool:
        er_distances = pool.map(calc_syn_distance, args)

    with open(figfolder / f'{er_type}_syn_distances.pkl', 'wb') as f:
        pickle.dump(er_distances, f)


if __name__=="__main__":

    figfolder = pathlib.Path('/media/mplitt/SSD_storage/fig_scratch/EL_connectomics/syn_distances')
    figfolder.mkdir(parents=True, exist_ok=True)

    c = nmp.connectomics.npt_client()


    er_types = ('ER4d', 'ER4m', 
                'ER2_a', 'ER2_b', 'ER2_c', 'ER2_d',
                'ER3w_b', 'ER3a_c', 'ER3p_a', 'ER3w_a', 'ER3a_b', 'ER3p_b', 'ER3d_c', 'ER3d_a',
                'ER3d_d', 'ER3d_b', 'ER3a_a', 'ER3a_d', 'ER3m', 
                'ER1_b', 'ER1_a')
    for type in er_types:
        print(type)
        run_er_type(type)

    er4d_syns, er4ds = get_er_syn()
    # # er4ds, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type = 'ER4d'))

    # args = []
    # for id in er4ds['bodyId'].iloc[:1].tolist():
    #     mask = (er4d_syns['type']=='epg_pre') * (er4d_syns['bodyId_pre']==id)
    #     mask = mask | ((er4d_syns['type'].str.contains('post')) * (er4d_syns['bodyId_post']==id))
    #     cell_syns = er4d_syns.loc[mask,:]
    #     args.append((id, cell_syns))
        

    # with Pool(processes=1) as pool:
    #     er4d_distances = pool.map(calc_syn_distance, args)

    # with open(figfolder / 'er4d_syn_distances.pkl', 'wb') as f:
    #     pickle.dump(er4d_distances, f)