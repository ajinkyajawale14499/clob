find_package(pybind11 REQUIRED)

pybind11_add_module(clob_py pyclob.cpp)
target_link_libraries(clob_py PRIVATE clob::matching)
target_compile_features(clob_py PRIVATE cxx_std_23)

# pybind11 generates a .cpython-<ver>-<plat>.so; for Python import we want it
# importable as `clob_py`. The default output path is fine; tests add the
# build dir to sys.path via conftest.py.
