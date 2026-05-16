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
        # W6+: quill (async logging)
        # W10+: onnxruntime, hdrhistogram_c

    def layout(self):
        cmake_layout(self)
