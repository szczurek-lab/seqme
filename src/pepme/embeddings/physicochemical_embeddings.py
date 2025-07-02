from pepme.properties.physicochemical_properties import PhysicochemicalPropertyAggregator

def physicochemical_embeddings(sequences, scaling = "standard"):
    return PhysicochemicalPropertyAggregator().compute(sequences, scaling=scaling)