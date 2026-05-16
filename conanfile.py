from conan import ConanFile
from conan.tools.cmake import cmake_layout


class ClobConan(ConanFile):
    name = "clob"
    version = "0.1.0"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def requirements(self):
        # Latest stable on Conan Center as of 2026-05-16.
        self.requires("catch2/3.7.1")
        self.requires("rapidcheck/cci.20231215")
        # W9: pybind11 for Engine Python bindings.
        self.requires("pybind11/3.0.1")
        # W10: onnxruntime + hdrhistogram-c (note hyphen) + nlohmann_json for
        # microprice LUT JSON parsing on the C++ side.
        self.requires("onnxruntime/1.24.4")
        self.requires("hdrhistogram-c/0.11.8")
        self.requires("nlohmann_json/3.11.3")

    def layout(self):
        cmake_layout(self)
