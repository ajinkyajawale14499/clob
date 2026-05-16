find_package(hdr_histogram REQUIRED)

add_executable(bench-scoring bench_scoring.cpp)
# Conan target is `hdr_histogram::hdr_histogram_static` (note _static suffix).
target_link_libraries(bench-scoring
    PRIVATE clob::matching hdr_histogram::hdr_histogram_static
)
target_compile_features(bench-scoring PRIVATE cxx_std_23)
