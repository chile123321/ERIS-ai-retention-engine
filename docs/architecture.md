# Architecture

Configuration defines the data contract, feature sets, models, and evaluation policy. Scripts call production modules in `src/eris_ml`. Training will persist one versioned bundle containing preprocessing plus model so serving uses exactly the same transformations. FastAPI will load an approved bundle during application lifespan in a later milestone; imports must remain side-effect free.
