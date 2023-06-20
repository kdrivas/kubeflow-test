from kfp import dsl, compiler
from kfp.client import Client

@dsl.component(
    packages_to_install=["numpy", "pandas"]
)
def hello_func(a: str) -> int:
    import numpy as np
    import pandas as pd
    print("Leyendo un numpy")
    d = np.array([0,1])
    print("Leyendo un df")
    f = pd.DataFrame({"col1": [1,2,3], "col2": [2,3,4]})
    print(len(a))
    return len(a)

@dsl.pipeline
def my_pipeline(recipient: str) -> int:
    
    res = hello_func(a=recipient)

    return res.output

compiler.Compiler().compile(
    my_pipeline,
    "my_pipeline.yaml"
)

client = Client(host='http://localhost:8080')
run = client.create_run_from_pipeline_package(
    'my_pipeline.yaml',
    arguments={
        'recipient': 'World',
    },
)