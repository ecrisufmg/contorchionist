#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "core_ap_melspectrogram.h"
#include "core_ap_rmsoverlap.h"
#include "core_util_conversions.h"
#include "core_util_windowing.h"
#include "core_util_normalizations.h"

namespace py = pybind11;
using namespace contorchionist::core::ap_melspectrogram;
using namespace contorchionist::core::ap_rmsoverlap;
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

    // ENUM 6: WindowAlignment
    py::enum_<contorchionist::core::util_windowing::Alignment>(m, "WindowAlignment")
        .value("LEFT", contorchionist::core::util_windowing::Alignment::LEFT)
        .value("CENTER", contorchionist::core::util_windowing::Alignment::CENTER)
        .value("RIGHT", contorchionist::core::util_windowing::Alignment::RIGHT)
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

    // --- RMSOverlap ---
    using RMSOverlapFloat = RMSOverlap<float>;

    py::enum_<RMSOverlapFloat::NormalizationType>(m, "RMSOverlapNormalizationType")
        .value("WINDOW_OVERLAP_RMS", RMSOverlapFloat::NormalizationType::WINDOW_OVERLAP_RMS)
        .value("WINDOW_OVERLAP_MEAN", RMSOverlapFloat::NormalizationType::WINDOW_OVERLAP_MEAN)
        .value("WINDOW_OVERLAP_VALS", RMSOverlapFloat::NormalizationType::WINDOW_OVERLAP_VALS)
        .value("OVERLAP_INVERSE", RMSOverlapFloat::NormalizationType::OVERLAP_INVERSE)
        .value("FIXED_MULTIPLIER", RMSOverlapFloat::NormalizationType::FIXED_MULTIPLIER)
        .value("NONE", RMSOverlapFloat::NormalizationType::NONE)
        .export_values();

    py::class_<RMSOverlapFloat>(m, "RMSOverlap")
        .def(py::init<int, int, contorchionist::core::util_windowing::Type, float, contorchionist::core::util_windowing::Alignment, RMSOverlapFloat::NormalizationType, float, int, bool>(),
             py::arg("initialWindowSize") = 1024,
             py::arg("initialHopSize") = 512,
             py::arg("initialWinType") = contorchionist::core::util_windowing::Type::HANN,
             py::arg("zeroPaddingFactor") = 0.0f,
             py::arg("initialWinAlign") = contorchionist::core::util_windowing::Alignment::CENTER,
             py::arg("initialNormType") = RMSOverlapFloat::NormalizationType::WINDOW_OVERLAP_RMS,
             py::arg("fixedNormMultiplier") = 1.0f,
             py::arg("initialBlockSize") = 64,
             py::arg("verbose") = false,
             "RMSOverlap Constructor"
        )
        .def("post_input_data", [](RMSOverlapFloat& self, py::array_t<float, py::array::c_style | py::array::forcecast> input) {
            self.post_input_data(input.data(), input.size());
        }, py::arg("input"), "Post input audio data to the internal circular buffer.")

        .def("process", [](RMSOverlapFloat& self, int inputBlockSize) {
            auto result = py::array_t<float>(inputBlockSize);
            py::buffer_info buf = result.request();
            float* ptr = static_cast<float*>(buf.ptr);
            self.process(nullptr, ptr, inputBlockSize);
            return result;
        }, py::arg("inputBlockSize"), "Process data from the circular buffer and get a block of RMS samples.")

        // Setters
        .def("set_window_size", &RMSOverlapFloat::setWindowSize, py::arg("newWindowSize"))
        .def("set_hop_size", &RMSOverlapFloat::setHopSize, py::arg("newHopSize"))
        .def("set_window_type", &RMSOverlapFloat::setWindowType, py::arg("newType"))
        .def("set_block_size", &RMSOverlapFloat::setBlockSize, py::arg("newBlockSize"))
        .def("set_normalization", &RMSOverlapFloat::setNormalization, py::arg("newType"), py::arg("fixedMultiplier") = 1.0f)
        .def("set_zero_padding", &RMSOverlapFloat::setZeroPadding, py::arg("factor"), py::arg("alignment"))

        // Getters
        .def("get_window_size", &RMSOverlapFloat::getWindowSize)
        .def("get_hop_size", &RMSOverlapFloat::getHopSize)
        .def("get_block_size", &RMSOverlapFloat::getBlockSize)
        .def("get_window_type", &RMSOverlapFloat::getWindowType)
        .def("get_normalization_type", &RMSOverlapFloat::getNormalizationType)

        // Other methods
        .def("reset", &RMSOverlapFloat::reset, "Reset the processor's internal state.");
}
