#include "m_pd.h"
#include "../../../core/include/core_util_circbuffer.h" // Inclui a classe CircularBuffer robusta
#include "../utils/include/pd_arg_parser.h"
#include <vector>
#include <algorithm> // para std::max

// Apelido para a classe C++ para melhorar a legibilidade
using CircularBufferF = contorchionist::core::util_circbuffer::CircularBuffer<float>;

// =====================================================================================
// DEFINIÇÃO DA ESTRUTURA DO OBJETO PD
// =====================================================================================
typedef struct _torch_circbuffer_test_tilde {
    t_object x_obj;
    t_float x_f; // Necessário para CLASS_MAINSIGNALIN

    // --- Membros C++ gerenciados por ponteiros ---
    CircularBufferF* buffer_;
    std::vector<float>* read_delays_samples_;
    std::vector<long>* idx_read_points_; // Índices para visualização

    // --- Índices de estado para visualização ---
    long idx_write_point_;

    // --- Membros de estado do Pd ---
    long sample_rate_;
    bool verbose_;
    long buffer_size_samples_;
    int num_read_points_;

    // --- Recursos do Pd (outlets) ---
    t_outlet** signal_outlets_;
    t_outlet* info_outlet_;
    
} t_torch_circbuffer_test_tilde;

// Ponteiro para a classe do objeto
static t_class *torch_circbuffer_test_tilde_class;

// =====================================================================================
// FUNÇÃO DE PROCESSAMENTO DE ÁUDIO (PERFORM)
// =====================================================================================
static t_int *torch_circbuffer_test_tilde_perform(t_int *w) {
    t_torch_circbuffer_test_tilde *x = (t_torch_circbuffer_test_tilde *)(w[1]);
    t_sample *in = (t_sample *)(w[2]);
    int n = (int)(w[3 + x->num_read_points_]);

    // Verificação de segurança: se algum recurso essencial não existir, encerre.
    if (!x || !x->buffer_ || !x->read_delays_samples_ || !x->idx_read_points_) {
        return (w + 4 + x->num_read_points_);
    }

    // 1. Processamento de Áudio: Escrita e Leitura
    x->buffer_->write_overwrite(in, n);
    for (int i = 0; i < x->num_read_points_; i++) {
        t_sample *out = (t_sample *)(w[3 + i]);
        size_t delay_samples = static_cast<size_t>(std::max(0.0f, (*x->read_delays_samples_)[i]));
        x->buffer_->peek_with_delay_and_fill(out, n, delay_samples);
    }
    
    // 2. Atualização dos Índices de Visualização
    x->idx_write_point_ = (x->idx_write_point_ + n) % x->buffer_size_samples_;
    for (int i = 0; i < x->num_read_points_; i++) {
        (*x->idx_read_points_)[i] = ((*x->idx_read_points_)[i] + n) % x->buffer_size_samples_;
    }

    // 3. Saída de Informações de Depuração
    if (x->info_outlet_) {
        // Envia o ponto de escrita
        t_atom write_list[4];
        SETSYMBOL(&write_list[0], gensym("write_point"));
        SETFLOAT(&write_list[1], 0.0f); // Índice do ponto (sempre 0)
        SETFLOAT(&write_list[2], static_cast<t_float>(x->idx_write_point_));
        SETFLOAT(&write_list[3], static_cast<t_float>(x->idx_write_point_) / x->sample_rate_);
        outlet_anything(x->info_outlet_, gensym("list"), 4, write_list);

        // Envia os pontos de leitura
        for (int i = 0; i < x->num_read_points_; i++) {
            t_atom read_list[4];
            SETSYMBOL(&read_list[0], gensym("read_point"));
            SETFLOAT(&read_list[1], static_cast<t_float>(i));
            SETFLOAT(&read_list[2], static_cast<t_float>((*x->idx_read_points_)[i]));
            SETFLOAT(&read_list[3], static_cast<t_float>((*x->idx_read_points_)[i]) / x->sample_rate_);
            outlet_anything(x->info_outlet_, gensym("list"), 4, read_list);
        }
    }

    // Retorna o ponteiro para o próximo conjunto de argumentos na cadeia DSP
    return (w + 4 + x->num_read_points_);
}

// =====================================================================================
// CONFIGURAÇÃO DO DSP (QUANDO O ÁUDIO É LIGADO)
// =====================================================================================
static void torch_circbuffer_test_tilde_dsp(t_torch_circbuffer_test_tilde *x, t_signal **sp) {
    if (!x || !x->buffer_) {
        pd_error(x ? &x->x_obj : NULL, "torch.circbuffer_test~: objeto ou buffer inválido no _dsp");
        return;
    }

    x->sample_rate_ = sp[0]->s_sr;
    
    // AÇÃO CRÍTICA DE RESET: Garante um estado limpo a cada ativação do DSP
    x->buffer_->clear();
    x->idx_write_point_ = 0;
    for (int i = 0; i < x->num_read_points_; i++) {
        long delay_samples = static_cast<long>((*x->read_delays_samples_)[i]);
        (*x->idx_read_points_)[i] = (x->buffer_size_samples_ - delay_samples + x->buffer_size_samples_) % x->buffer_size_samples_;
    }
    
    // Montagem dos argumentos para dsp_addv de forma segura
    int n_args_total = 3 + x->num_read_points_;
    std::vector<t_int> dsp_vec(n_args_total);

    dsp_vec[0] = (t_int)x;
    dsp_vec[1] = (t_int)sp[0]->s_vec;
    for (int i = 0; i < x->num_read_points_; i++) {
        dsp_vec[2 + i] = (t_int)sp[1 + i]->s_vec;
    }
    dsp_vec[2 + x->num_read_points_] = (t_int)sp[0]->s_n;

    dsp_addv(torch_circbuffer_test_tilde_perform, n_args_total, dsp_vec.data());

    if (x->verbose_) {
        post("torch.circbuffer_test~: dsp reconfigurado (SR: %ld)", x->sample_rate_);
    }
}

// =====================================================================================
// DESTRUTOR (QUANDO O OBJETO É DELETADO)
// =====================================================================================
static void torch_circbuffer_test_tilde_free(t_torch_circbuffer_test_tilde *x) {
    // Libera a memória na ordem inversa da alocação
    delete x->buffer_;
    delete x->read_delays_samples_;
    delete x->idx_read_points_;
    delete[] x->signal_outlets_;
}

// =====================================================================================
// CONSTRUTOR (QUANDO O OBJETO É CRIADO)
// =====================================================================================
static void *torch_circbuffer_test_tilde_new(t_symbol *s, int argc, t_atom *argv) {
    t_torch_circbuffer_test_tilde *x = (t_torch_circbuffer_test_tilde *)pd_new(torch_circbuffer_test_tilde_class);
    if (!x) return nullptr;

    // Inicializa todos os ponteiros e membros para um estado seguro
    x->buffer_ = nullptr;
    x->read_delays_samples_ = nullptr;
    x->idx_read_points_ = nullptr;
    x->signal_outlets_ = nullptr;
    x->info_outlet_ = nullptr;
    x->sample_rate_ = sys_getsr();
    x->verbose_ = false;
    x->buffer_size_samples_ = x->sample_rate_;
    x->num_read_points_ = 2;
    x->idx_write_point_ = 0;

    // Analisa os argumentos de criação
    pd_utils::ArgParser parser(argc, argv, &x->x_obj);
    x->verbose_ = parser.has_flag("verbose") || parser.has_flag("v");
    x->buffer_size_samples_ = static_cast<long>(parser.get_float("bufsize", 1.0f) * x->sample_rate_);
    x->num_read_points_ = static_cast<int>(parser.get_float("num_readpoints", 2.0f));

    // Validação de parâmetros
    if (x->buffer_size_samples_ <= 0) x->buffer_size_samples_ = x->sample_rate_;
    if (x->num_read_points_ < 1) x->num_read_points_ = 1;
    if (x->num_read_points_ > 64) x->num_read_points_ = 64;

    // Alocação de memória (bloco try/catch para segurança)
    try {
        x->buffer_ = new CircularBufferF(x->buffer_size_samples_);
        x->read_delays_samples_ = new std::vector<float>(x->num_read_points_, 0.0f);
        x->idx_read_points_ = new std::vector<long>(x->num_read_points_, 0);
        
        x->signal_outlets_ = new t_outlet*[x->num_read_points_];
        for (int i = 0; i < x->num_read_points_; i++) {
            x->signal_outlets_[i] = outlet_new(&x->x_obj, &s_signal);
        }
        
        x->info_outlet_ = outlet_new(&x->x_obj, &s_list);
        
        // Inicializa delays e posições iniciais dos índices de leitura
        for (int i = 0; i < x->num_read_points_; i++) {
            float delay_samples = (x->buffer_size_samples_ / (float)x->num_read_points_) * i;
            (*x->read_delays_samples_)[i] = delay_samples;
            (*x->idx_read_points_)[i] = (x->buffer_size_samples_ - static_cast<long>(delay_samples) + x->buffer_size_samples_) % x->buffer_size_samples_;
        }
        
        if (x->verbose_) {
            post("torch.circbuffer_test~: criado com buffer de %ld amostras e %d pontos de leitura", x->buffer_size_samples_, x->num_read_points_);
        }
        
    } catch (const std::exception& e) {
        pd_error(&x->x_obj, "torch.circbuffer_test~: falha na alocação de memória: %s", e.what());
        torch_circbuffer_test_tilde_free(x);
        return nullptr;
    }

    return (void *)x;
}

// =====================================================================================
// MÉTODOS ADICIONAIS (para mensagens)
// =====================================================================================
static void torch_circbuffer_test_tilde_delay(t_torch_circbuffer_test_tilde *x, t_symbol *s, int argc, t_atom *argv) {
    if (argc < 2) {
        pd_error(&x->x_obj, "torch.circbuffer_test~: uso: delay <índice_leitura> <delay_amostras>");
        return;
    }
    
    int read_point = atom_getint(&argv[0]);
    float delay_samples = atom_getfloat(&argv[1]);
    
    if (read_point < 0 || read_point >= x->num_read_points_) {
        pd_error(&x->x_obj, "torch.circbuffer_test~: índice de leitura inválido: %d (deve ser 0 a %d)", read_point, x->num_read_points_ - 1);
        return;
    }
    
    (*x->read_delays_samples_)[read_point] = delay_samples;
    if (x->verbose_) {
        post("torch.circbuffer_test~: ponto de leitura %d configurado com delay de %.1f amostras", read_point, delay_samples);
    }
}

// Outros métodos de mensagem como _info, _status, _verbose podem ser adicionados aqui...

// =====================================================================================
// FUNÇÃO DE SETUP (REGISTRA A CLASSE NO PD)
// =====================================================================================
extern "C" void setup_torch0x2ecircbuffer_test_tilde(void) {
    torch_circbuffer_test_tilde_class = class_new(
        gensym("torch.circbuffer_test~"),
        (t_newmethod)torch_circbuffer_test_tilde_new,
        (t_method)torch_circbuffer_test_tilde_free,
        sizeof(t_torch_circbuffer_test_tilde),
        CLASS_DEFAULT,
        A_GIMME, 0
    );

    class_addmethod(torch_circbuffer_test_tilde_class,
                    (t_method)torch_circbuffer_test_tilde_dsp, gensym("dsp"), A_CANT, 0);
                    
    class_addmethod(torch_circbuffer_test_tilde_class,
                    (t_method)torch_circbuffer_test_tilde_delay, gensym("delay"), A_GIMME, 0);
                    
    // Adicione aqui outros class_addmethod para _info, _status, etc., se desejar.

    CLASS_MAINSIGNALIN(torch_circbuffer_test_tilde_class, t_torch_circbuffer_test_tilde, x_f);
}