from typing import Union, Optional, List, Dict

import numpy as np
import numba as nb
import awkward as ak
import vector
vector.register_awkward()

from quickstats import DescriptiveEnum

class Momentum4DFormat(DescriptiveEnum):
    PtEtaPhiM = (0, "Four momentum in the form of (pT, eta, phi, M)")
    PtEtaPhiE = (1, "Four momentum in the form of (pT, eta, phi, E)")
    PxPyPzM = (2, "Four momentum in the form of (px, py, pz, M)")
    PxPyPzE = (3, "Four momentum in the form of (px, py, pz, E)")


class Momentum4DArrayBuilder:

    @property
    def fmt(self) -> Momentum4DFormat:
        return self._fmt

    @fmt.setter
    def fmt(self, value):
        self._fmt = Momentum4DFormat.parse(value)

    def __init__(self, fmt:Union[Momentum4DFormat, str]="PtEtaPhiM"):
        self.fmt = fmt
    
    @staticmethod
    @nb.njit
    def build_PtEtaPhiM_array(builder, events):
        for event in events:
            builder.begin_list()
            for particle in event:
                builder.begin_record("Momentum4D")
                builder.field("pt").real(particle[0])
                builder.field("eta").real(particle[1])
                builder.field("phi").real(particle[2])
                builder.field("m").real(particle[3])
                builder.end_record()
            builder.end_list()
        return builder
        
    @staticmethod
    @nb.njit
    def build_PtEtaPhiE_array(builder, events):
        for event in events:
            builder.begin_list()
            for particle in event:
                builder.begin_record("Momentum4D")
                builder.field("pt").real(particle[0])
                builder.field("eta").real(particle[1])
                builder.field("phi").real(particle[2])
                builder.field("e").real(particle[3])
                builder.end_record()
            builder.end_list()
        return builder

    @staticmethod
    @nb.njit
    def build_PxPyPzM_array(builder, events):
        for event in events:
            builder.begin_list()
            for particle in event:
                builder.begin_record("Momentum4D")
                builder.field("px").real(particle[0])
                builder.field("py").real(particle[1])
                builder.field("pz").real(particle[2])
                builder.field("m").real(particle[3])
                builder.end_record()
            builder.end_list()
        return builder

    @staticmethod
    @nb.njit
    def build_PxPyPzE_array(builder, events):
        for event in events:
            builder.begin_list()
            for particle in event:
                builder.begin_record("Momentum4D")
                builder.field("px").real(particle[0])
                builder.field("py").real(particle[1])
                builder.field("pz").real(particle[2])
                builder.field("e").real(particle[3])
                builder.end_record()
            builder.end_list()
        return builder
        
    @staticmethod
    def get_array_from_dict(data:Dict[str, np.ndarray]):
        """
        e.g. data = {
                "pt"  : <pt_arrays>,
                "eta" : <eta_arrays>,
                "phi" : <phi_arrays>,
                "m"   : <m_arrays>
              }
        """
        return ak.zip(data, with_name="Momentum4D")

    def get_array_from_list(self, data:List[np.ndarray]):
        build_fn_map = {
            Momentum4DFormat.PtEtaPhiM : self.build_PtEtaPhiM_array,
            Momentum4DFormat.PtEtaPhiE : self.build_PtEtaPhiE_array,
            Momentum4DFormat.PxPyPzM   : self.build_PxPyPzM_array,
            Momentum4DFormat.PxPyPzE   : self.build_PxPyPzE_array
        }
        build_fn = build_fn_map[self.fmt]
        builder = ak.ArrayBuilder()
        nb_list = nb.typed.List(data)
        return build_fn(builder, nb_list).snapshot()