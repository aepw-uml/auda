from auda.utils.pipeline import add_kind, scan_package

scan_package(__path__, __name__)

add_kind('DS', 'dataset')
add_kind('MD', 'model')
