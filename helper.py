import numpy as np



# ------------------------ Some helper functions for this project -------------------------


def time_elapsed(t0,t):
    """
    print time elapsed between t0 and t, where t0 and t are time.time() instances
    """
    delta_t = t-t0
    
    t_mins = round(delta_t/60, 2)
    
    return t_mins


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



def diffuse(data, all_features, noise_std=1, apply_features=None):
    """
    Given input data (e.g., output of preprocess.get_data) and keyword arguments,
    diffuse high quality input to lower quality by adding Gaussian noise to specified features.

    data: numpy array of shape [INPUT_SIZE, NUM_FEATURES, NUM_CONSTITUENTS] or [INPUT_SIZE, NUM_FEATURES]
    noise_std: desired noise standard deviation for the diffusion
    all_features: list of all features (ordered wrt data) of shape [NUM_FEATURES]
    apply_features: list of features to apply diffusion on
    """
    # Create a copy of the data to avoid inplace contamination
    data_copy = np.copy(data)

    # Next, replace zeros (which aren't physically relevant as they just correspond to missing data) with NaN
    data_copy = np.where(data_copy == 0, np.nan, data_copy)
    
    use_features = all_features if apply_features is None else apply_features

    for f in use_features:
        f_idx = np.where(all_features == f)[0][0]

        if data_copy.ndim == 3:
            data_copy[:, f_idx, :] += np.random.normal(loc=0.0, scale=noise_std, size=data_copy[:, f_idx, :].shape)

        elif data_copy.ndim == 2:
            data_copy[:, f_idx] += np.random.normal(loc=0.0, scale=noise_std, size=data_copy[:, f_idx].shape)

        else:
            raise ValueError("diffuse doesn't apply to high-level data. Try jet or constituents.")

    # Finally, add the zeros back in to correctly handle missing values
    data_copy = np.nan_to_num(data_copy)

    return data_copy
