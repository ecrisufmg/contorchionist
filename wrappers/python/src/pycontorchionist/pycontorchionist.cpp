#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "core_ap_melspectrogram.h"
#include "core_util_conversions.h"
#include "core_util_windowing.h"
#include "core_util_normalizations.h"

namespace py = pybind11;
using namespace contorchionist::core::ap_melspectrogram;
using namespace contorchionist::core::util_conversions;
using namespace contorchionist::core::util_windowing;
using namespace contorchionist::core::util_normalizations;

PYBIND11_MODULE(pycontorchionist, m) {
    m.doc() = "Contorchionist Audio Processing Library";

    // ENUM 1: MelNormMode
    py::enum_<MelNormMode>(m, "MelNormMode")
        .value("NONE", MelNormMode::NONE)
        .value("ENERGY_POWER", MelNormMode::ENERGY_POWER)
        .value("MAGNITUDE_SUM", MelNormMode::MAGNITUDE_SUM)
        .export_values();

    // ENUM 2: SpectrumDataFormat
    py::enum_<SpectrumDataFormat>(m, "SpectrumDataFormat")
        .value("COMPLEX", SpectrumDataFormat::COMPLEX)
        .value("MAGPHASE", SpectrumDataFormat::MAGPHASE)
        .value("POWERPHASE", SpectrumDataFormat::POWERPHASE)
        .value("DBPHASE", SpectrumDataFormat::DBPHASE)
        .export_values();

    // ENUM 3: MelFormulaType
    py::enum_<MelFormulaType>(m, "MelFormulaType")
        .value("SLANEY", MelFormulaType::SLANEY)
        .value("HTK", MelFormulaType::HTK)
        .value("CALC2", MelFormulaType::CALC2)
        .export_values();

    // ENUM 4: WindowType
    py::enum_<contorchionist::core::util_windowing::Type>(m, "WindowType")
        .value("RECTANGULAR", contorchionist::core::util_windowing::Type::RECTANGULAR)
        .value("HANN", contorchionist::core::util_windowing::Type::HANN)
        .value("HAMMING", contorchionist::core::util_windowing::Type::HAMMING)
        .value("BLACKMAN", contorchionist::core::util_windowing::Type::BLACKMAN)
        .value("BARTLETT", contorchionist::core::util_windowing::Type::BARTLETT)
        .value("COSINE", contorchionist::core::util_windowing::Type::COSINE)
        .export_values();

    // ENUM 5: NormalizationType
    py::enum_<contorchionist::core::util_normalizations::NormalizationType>(m, "NormalizationType")
        .value("NONE", contorchionist::core::util_normalizations::NormalizationType::NONE)
        .value("BACKWARD", contorchionist::core::util_normalizations::NormalizationType::BACKWARD)
        .value("FORWARD", contorchionist::core::util_normalizations::NormalizationType::FORWARD)
        .value("ORTHO", contorchionist::core::util_normalizations::NormalizationType::ORTHO)
        .value("WINDOW", contorchionist::core::util_normalizations::NormalizationType::WINDOW)
        .value("POWER", contorchionist::core::util_normalizations::NormalizationType::POWER)
        .value("MAGNITUDE", contorchionist::core::util_normalizations::NormalizationType::MAGNITUDE)
        .export_values();

    //     HELPER FUNCTIONS para MelNormMode
    m.def("mel_norm_mode_to_string", &mel_norm_mode_to_string, 
          "Convert MelNormMode enum to string");
    
    m.def("string_to_mel_norm_mode", &string_to_mel_norm_mode, 
          "Convert string to MelNormMode enum");

    //     HELPER FUNCTIONS para SpectrumDataFormat
    m.def("spectrum_data_format_to_string", &spectrum_data_format_to_string, 
          "Convert SpectrumDataFormat enum to string");
    
    m.def("string_to_spectrum_data_format", &string_to_spectrum_data_format, 
          "Convert string to SpectrumDataFormat enum");

    //     CLASSE MelSpectrogramProcessor
    py::class_<MelSpectrogramProcessor<float>>(m, "MelSpectrogramProcessor")
        .def(py::init<>(), "Default constructor with standard parameters")
        // Construtor completo com parâmetros posicionais
        .def(py::init<int, int, int, contorchionist::core::util_windowing::Type, 
                      contorchionist::core::util_normalizations::NormalizationType,
                      contorchionist::core::util_conversions::SpectrumDataFormat,
                      float, int, float, float, contorchionist::core::util_conversions::MelFormulaType,
                      const std::string&, MelNormMode, torch::Device, bool>(),
             "Full constructor")
        // ...outros métodos...
        .def("process",
            [](MelSpectrogramProcessor<float>& self, py::array_t<float, py::array::c_style | py::array::forcecast> input) -> py::object {
                std::vector<float> out1, out2;
                bool ok = self.process(input.data(), input.size(), out1, out2);
                if (!ok) return py::none();
                return py::array_t<float>(out1.size(), out1.data());
            },
            py::arg("input"),
            R"pbdoc(
                Processa um bloco de áudio e retorna o vetor de mels.
                Retorna None se não houver frame pronto.
            )pbdoc"
        )

        // Getters
        .def("get_sample_rate", &MelSpectrogramProcessor<float>::get_sample_rate)
        .def("get_n_mels", &MelSpectrogramProcessor<float>::get_n_mels)
        .def("get_n_fft", &MelSpectrogramProcessor<float>::get_n_fft)
        .def("get_hop_length", &MelSpectrogramProcessor<float>::get_hop_length)
        
        // Setters
        .def("set_sample_rate", &MelSpectrogramProcessor<float>::set_sample_rate)
        .def("set_n_mels", &MelSpectrogramProcessor<float>::set_n_mels)
        .def("set_n_fft", &MelSpectrogramProcessor<float>::set_n_fft)
        .def("set_hop_length", &MelSpectrogramProcessor<float>::set_hop_length)
        
        // Métodos
        .def("clear_buffer", &MelSpectrogramProcessor<float>::clear_buffer)
        ;
}
