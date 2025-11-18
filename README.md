# contorchionist

*contorchionist* is a C++ project that leverages the power of `libtorch` (the C++ backend of PyTorch) to implement audio processing, machine listening, and machine learning functionalities. The main goal is to provide a unified framework that can be seamlessly integrated into various real-time and non-real-time environments, such as Pure Data (PD), Max, and Python.


Inspired by existing tools like [FluCoMa](https://flucoma.org), [timbreID](https://github.com/wbrent/timbreIDLib/), [SCMIR](https://composerprogrammer.com/code.html), [nn~](https://github.com/acids-ircam/nn_tilde), and others, *contorchionist* aims to create a flexible and nomadic library based on [libtorch](https://pytorch.org/cppdocs/) for exploring machine listening/learning techniques in diverse languages/environments, OSes, and time contexts.

DISCLAIMER: The project is currently in its early stages of development. For now, our main focus is to implement some few core functionalities and have them working (i.e., producing the same results) across different languages and environments. Because of that, we are not providing pre-compiled binaries yet (see the Compilation section below if you want to try it out). For more ready-to-use tools, please check the tools mentioned above.

## Compilation

At this moment, we do not provide pre-compiled binaries. To use the library, you need to compile it for your specific environment and language by modifying the options in the main `CMakeLists.txt` file.

### Building the Project

The project uses CMake as its build system. To configure and build:

```bash
# Configure the project
cmake -S . -B build

# Build all targets
cmake --build build
```

### Build Options

You can enable/disable specific wrappers by setting the following options in `CMakeLists.txt`:

- `BUILD_PD_WRAPPER` - Build the Pure Data wrapper (default: ON)
- `BUILD_MAX_WRAPPER` - Build the Max/MSP wrapper (default: ON)
- `BUILD_SC_WRAPPER` - Build the SuperCollider wrapper (default: ON)
- `BUILD_PYTHON_WRAPPER` - Build the Python wrapper (default: ON)
- `BUILD_TESTS` - Build tests for the project (default: OFF)

## Installation

### Pure Data and PlugData

After building the project, you can install the externals to Pure Data and PlugData using the custom install target:

```bash
cmake --build build --target install-puredata
```

This will automatically copy the distribution folder to:
- **Pure Data**: `~/Documents/Pd/externals/conTorchinistSN/`
- **PlugData**: `~/Documents/plugdata/Externals/conTorchinistSN/`

The installation will only proceed if the target directories exist. All necessary files (externals, dynamic libraries, help patches, and resources) will be copied.

Alternatively, you can use the standard CMake install command:

```bash
cmake --install build
```

## Acknowledgments

This project is developed at [ECrIS - Espaço de Criação de Investigação Sonora](https://ecris.cc/), a laboratory of the [School of Music of the Federal University of Minas Gerais (UFMG)](https://www.musica.ufmg.br/), with support from CNPq and FAPEMIG.

### Development Team

*   **José Henrique Padovani** (Professor, School of Music, UFMG)
*   **Vinícius César de Oliveira** (PhD Candidate, PPGMUS, Institute of Arts, UNICAMP)
