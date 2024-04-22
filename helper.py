import numpy as np



# ------------------------ Some helper functions for this project -------------------------



def human_feature(feature):
    """
    Given a feature (string) as presented in the ATLAS Top Tagging dataset,
    internally compare it to a feature mapping dictionary and return the 
    human-readable feature

    str --> str
    """

    feature_dict = {
        'fjet_C2': None,
        'fjet_D2': None,
        'fjet_ECF1': None,
        'fjet_ECF2': None,
        'fjet_ECF3': None,
        'fjet_L2': None,
        'fjet_L3': None,
        'fjet_Qw': None,
        'fjet_Split12': None,
        'fjet_Split23': None,
        'fjet_Tau1_wta': None,
        'fjet_Tau2_wta': None,
        'fjet_Tau3_wta': None,
        'fjet_Tau4_wta': None,
        'fjet_ThrustMaj': None,
        'fjet_eta': "jet pseudo-rapidity",
        'fjet_m': "jet mass",
        'fjet_phi': "jet azimuthal angle",
        'fjet_pt': "jet transverse momentum",
        'fjet_clus_pt': "constituent transverse momentum",
        'fjet_clus_eta': "constituent pseudo-rapidity",
        'fjet_clus_phi': "constituent azimuthal angle",
        'fjet_clus_E': "constituent energy",
    }

    return feature_dict[feature]



def features_by_attribute(attribute):
    """
    Given an attribute from 'jet', 'constituents', or 'high_level', return the 
    corresponding features
    """

    jet_keys = ['fjet_pt', 'fjet_eta', 'fjet_phi', 'fjet_m']
    const_keys = ['fjet_clus_pt', 'fjet_clus_eta', 'fjet_clus_phi', 'fjet_clus_E']
    hl_keys = ['fjet_C2', 'fjet_D2', 
               'fjet_ECF1', 'fjet_ECF2', 
               'fjet_ECF3', 'fjet_L2', 
               'fjet_L3', 'fjet_Qw', 
               'fjet_Split12', 'fjet_Split23', 
               'fjet_Tau1_wta', 'fjet_Tau2_wta', 
               'fjet_Tau3_wta', 'fjet_Tau4_wta', 'fjet_ThrustMaj']
    
    use_keys = jet_keys if attribute == 'jet' else (const_keys if attribute == 'constituents' else (hl_keys if attribute == 'high_level' else None))
    
    return use_keys





def diffuse(data, **kwargs):
    """
    Given input data (e.g. output of preprocess.extract) and keyword arguments,
    diffuse high quality input to lower quality
    """
    

    # TODO: Apply diffusion to input data


    reduced_data = None
    return reduced_data