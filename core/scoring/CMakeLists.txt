find_package(nlohmann_json REQUIRED)
find_package(onnxruntime REQUIRED)

# W10 task 6.2: FeatureState compiles independently of Scorer.
# Scorer (Task 6.3) adds Ort::Session dependency in a separate library.
add_library(clob_scoring STATIC
    feature_state.cpp
    scorer.cpp
)
target_include_directories(clob_scoring PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(clob_scoring
    PUBLIC clob::orderbook clob::events
    PRIVATE nlohmann_json::nlohmann_json onnxruntime::onnxruntime
)
target_compile_features(clob_scoring PUBLIC cxx_std_23)
add_library(clob::scoring ALIAS clob_scoring)
