from . import params

import neuprint as npt

def npt_client(webpage='neuprint.janelia.org', dataset='hemibrain:v1.2.1', token=params.NPT_TOKEN):
    """
    Create a neuprint client object with the specified parameters.
    
    Args:
        webpage (str): The URL of the neuprint server.
        dataset (str): The name of the dataset to use.
        token (str): The authentication token for the neuprint server.
        
    Returns:
        npt.Client: A neuprint client object.
    """
    return npt.Client(webpage, dataset=dataset, token=token)






