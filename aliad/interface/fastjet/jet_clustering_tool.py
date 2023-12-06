from typing import Union, Optional, List

import numpy as np
import numba as nb

# for jet clustering (old method)
import pyjet as pj
from pyjet import cluster, DTYPE_PTEPM

from aliad.interface.awkward import Momentum4DArrayBuilder

# for jet clustering (new method)
#import fastjet as fj

def get_jet_data(events):
    return [np.array([[j.pt, j.eta, j.phi, j.mass] for j in jets]) for jets in events]


class JetClusteringTool:

    def __init__(self, R:float, ptmin:float):
        self.R = R
        self.ptmin = ptmin
        self.array_builder = Momentum4DArrayBuilder("PtEtaPhiM")
        
    @staticmethod
    @nb.njit()
    def get_constituent_arrays_from_padded_data(event_data:np.ndarray,
                                                pad_size:Optional[int]=None):
        if pad_size is None:
            pad_size = event_data.shape[1] // 3
        pt_indices = np.arange(pad_size) * 3
        # check index of the first particle to have zero pt
        last_indices = np.argmax(event_data[:, pt_indices] == 0, axis=1)
        arrays = []
        nevent = event_data.shape[0]
        for i in range(nevent):
            nparticle = last_indices[i]
            array = np.zeros(nparticle, dtype=DTYPE_PTEPM)
            for j in range(nparticle):
                array[j]['pT']  = event_data[i][j * 3]
                array[j]['eta'] = event_data[i][j * 3 + 1]
                array[j]['phi'] = event_data[i][j * 3 + 2]
            arrays.append(array)
        return arrays

    def get_inclusive_jets(self, constituents:np.ndarray,
                           sort_by:Optional[str]=None):
        sequence = cluster(constituents, R=self.R, p=-1)
        jets = sequence.inclusive_jets(ptmin=self.ptmin)
        if sort_by is not None:
            if sort_by.lower() == "pt":
                jets = sorted(jets, key=lambda x: -x.pt)
            elif sort_by.lower() == "mass":
                jets = sorted(jets, key=lambda x: -x.mass)
            else:
                raise ValueError(f'unknown sort attribute: "{sort_by}"')
        return jets

    def get_inclusive_jets_array(self, constituents_arrays:np.ndarray,
                                 sort_by:Optional[str]=None):
        jets_array = [self.get_inclusive_jets(constituents, sort_by=sort_by) \
                      for constituents in constituents_arrays]
        return jets_array

    @staticmethod
    def to_awkward_jets_array(jets_array:List):
        array_builder = Momentum4DArrayBuilder("PtEtaPhiM")
        jet_data = [np.array([[j.pt, j.eta, j.phi, j.mass] for j in jets]) \
                    for jets in jets_array]
        return array_builder.get_array_from_list(jet_data)

    @staticmethod
    def to_awkward_constituents_array(jets_array:List, jet_index:int):
        array_builder = Momentum4DArrayBuilder("PtEtaPhiM")
        constituents_data = [jets[jet_index].constituents_array() \
                             if len(jets) > jet_index else np.array([], dtype=DTYPE_PTEPM) \
                             for jets in jets_array]
        return array_builder.get_array_from_list(constituents_data)